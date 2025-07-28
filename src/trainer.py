import os
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union
from torch.nn.utils.rnn import pad_sequence
import datasets
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available, is_rich_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
# from trl.models.utils import _ForwardRedirection
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

from trl.models import prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, selective_log_softmax
from client import get_other_express
import concurrent.futures
from datasets import Dataset, IterableDataset
from eval.chart.util import eval_one_chart
from eval.eval_metric import evaluate_single_sample

def more_express(question, answer, check_list, image_path, hints, gpu_id, tasks, num_threads=8):
    # 包装函数用于参数解包
    def process_item(args):
        q, a, c, i, h, t, gpu_id = args
        return get_other_express(q, a, c, i, h, gpu_id=gpu_id, expert_task=t)

    # 创建参数元组列表
    task_args = zip(question, answer, check_list, image_path, hints, tasks, [gpu_id]*len(question))

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        sents = list(executor.map(process_item, task_args))

    return sents

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_wandb_available():
    import wandb

def split_initial_context_final(text: str):
    text = text.lower()
    if ' answer: ' in text:
        ans = text.split(' answer: ')[-1]
        context = text.split(' answer: ')[0]
        ans = ans.strip('.')
    elif 'answer: ' in text:
        ans = text.split('answer: ')[-1]
        context = text.split('answer: ')[0]
        ans = ans.strip('.')
    else:
        context = text
        ans = ''
    return "", context, ans

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatSampler(Sampler):
    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    has = []
    if 'has_correct' in tensor_dict:
        has = tensor_dict['has_correct']
        del tensor_dict['has_correct']
    l1 = []
    for i in range(num_chunks):
        dt = {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        if len(has) > 0:
            dt['has_correct'] = has[i]
        l1.append(dt)

    return l1


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class MyGRPOTrainer(Trainer):
    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        processing_func: Optional[Callable[[dict], dict]] = None,
        attn_implementation: str = "flash_attention_2",
        end_template: str = None,
        start_template: str = None,
        prompt_template: str = None,
        is_sft: bool = False,
        task_name: str = None,
    ):
        self.end_template = end_template
        self.start_template = start_template
        self.is_sft = is_sft
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        self.ref_model = None
        self.task_name = task_name

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # processing_class.pad_token = processing_class.eos_token

        pad_token_id = processing_class.tokenizer.pad_token_id
        eos_token_id = processing_class.tokenizer.eos_token_id
        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Reference model
        self.beta = args.beta
        assert self.beta == 0

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Liger loss
        if self.use_liger_loss:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        }

        # Check if the effective batch size can be divided by the number of generations
        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )
        num_processes = self.accelerator.num_processes
        effective_batch_size = args.per_device_train_batch_size * num_processes * args.gradient_accumulation_steps
        possible_values = [
            n_gen for n_gen in range(2, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0
        ]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The effective train batch size ({num_processes} x {args.per_device_train_batch_size} x "
                f"{args.gradient_accumulation_steps}) must be evenly divisible by the number of generations per "
                f"prompt ({self.num_generations}). Given the current effective train batch size, the valid values for "
                f"the number of generations are: {possible_values}."
            )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)


        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            pad_token_id=processing_class.tokenizer.pad_token_id,
            bos_token_id=processing_class.tokenizer.bos_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            cache_implementation=args.cache_implementation,
            use_cache=False if self.args.gradient_checkpointing else True
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        self.new_prompt = prompt_template
        # self.new_prompt = get_good_format(self.accelerator.process_index)
        self.processing_func = processing_func

    def evaluate(self,
                 eval_dataset=None,
                 ignore_keys=None,
                 metric_key_prefix="eval"):
        # 1. 准备 eval dataloader
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_dataloader = self.accelerator.prepare(eval_dataloader)
        all_preds = []
        all_labels = []
        all_acc = []
        all_acc_ex = []
        # 2. 遍历每个 batch，生成并解码
        self.model.eval()

        # 2.1 调用 generate
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ):
                for step, batch in enumerate(eval_dataloader):
                    # 把 inputs 转到设备上
                    labels = []
                    prepare_input = []
                    answer_tp_list = []
                    for b in batch:
                        if 'chart' in self.task_name:
                            question = b['query']
                            prompt = (
                                    "For the question below, follow the following instructions:\n"
                                    + "-First output your step-by-step reasoning, then provide your answer beginning with “Answer:”.\n"
                                    + "-The answer should contain as few words as possible.\n"
                                    + "-Don't paraphrase or reformat the text you see in the image.\n"
                                    + "-Answer a binary question with Yes or No.\n"
                                    + "-When asked to give a numerical value, provide a number like 2 instead of Two.\n"
                                    + "-If the final answer has two or more items, provide it in the list format like [1, 2].\n"
                                    + "-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.\n"
                                    + "-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.\n"
                                    + "-Don't include any units in the answer.\n"
                                    + "-Do not include any full stops at the end of the answer.\n"
                                    + "-Try to include the full label from the graph when asked about an entity.\n"
                                    + "Question: "
                            )

                            question = prompt + question
                            b['question'] = question

                            # b['answer'] = b['label'][0]
                            b['answer'] = None
                            prepare_input.append(b)
                            labels.append(b['label'][0])
                        elif 'math' in self.task_name:
                            question = b['question'].strip()
                            answer = b['answer']
                            if question == '':
                                question = 'Answer the question shown in the image.'

                            b['question'] = question + ' Think step by step and then answer the question.'
                            b['answer'] = None
                            prepare_input.append(b)
                            labels.append(answer)
                        elif 'medical' in self.task_name:
                            question = b['question'].strip()
                            answer = b['answer']
                            tp = b['answer_type']

                            b['question'] = question + ' Think step by step and then answer the question.'
                            b['answer'] = None
                            b['answer_type'] = tp
                            prepare_input.append(b)
                            labels.append(answer)
                            answer_tp_list.append(tp)

                    prepare_input = self.processing_func(prepare_input)

                    inputs = {k: v.to(self.accelerator.device) for k, v in prepare_input.items()}

                    prompt_completion_ids = unwrapped_model.generate(**inputs,
                                                                     max_new_tokens=self.max_completion_length,
                                                                     do_sample=False)
                    # 2.2 解码成文本
                    prompt_ids = inputs["input_ids"]
                    prompt_length = prompt_ids.size(1)
                    prompt_ids = prompt_completion_ids[:, :prompt_length]
                    completion_ids = prompt_completion_ids[:, prompt_length:]

                    preds = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                    preds_ans = []
                    for completion in preds:
                        start, context, end = split_initial_context_final(completion)
                        preds_ans.append(end.lower())
                    preds = preds_ans
                    all_preds.extend(preds)
                    all_labels.extend(labels)

                    if 'chart' in self.task_name:
                        acc = [eval_one_chart(pred_answer, answer) for pred_answer, answer in zip(preds, labels)]
                    elif 'math' in self.task_name:
                        def eval_one_math(p, a):
                            return p.lower().strip() == a.lower().strip()
                        acc = [eval_one_math(pred_answer, answer) for pred_answer, answer in zip(preds, labels)]
                    elif 'medical' in self.task_name:
                        acc_open = []
                        acc_closed = []
                        for i in range(len(preds)):
                            answer_type = answer_tp_list[i]
                            ground_truth_answer = labels[i]
                            pred_answer = preds[i]
                            if answer_type == 'CLOSED':
                                score = evaluate_single_sample(ground_truth_answer, pred_answer, answer_type)[
                                    'yes/no accuracy']
                                acc_closed.append(score)
                            else:
                                score = evaluate_single_sample(ground_truth_answer, pred_answer, answer_type)[
                                    'recall']
                                acc_open.append(score)
                        acc = acc_open
                        all_acc_ex.extend(acc_closed)
                    all_acc.extend(acc)
                    # all_acc_show = self.accelerator.gather_for_metrics(
                    #     all_acc,
                    #     use_gather_object=True  # 因为它们是 Python list of str
                    # )
                    # if self.accelerator.is_main_process:
                    #     import numpy as np
                    #     all_acc_show = np.array(all_acc_show)
                    #     print(all_acc_show.mean())
            # 3. 计算指标
        all_acc_show = self.accelerator.gather_for_metrics(
            all_acc,
            use_gather_object=True  # 因为它们是 Python list of str
        )
        if 'medical' in self.task_name:
            all_acc_show_closed = self.accelerator.gather_for_metrics(
                all_acc_ex,
                use_gather_object=True  # 因为它们是 Python list of str
            )
        else:
            all_acc_show_closed = None
        metrics = {}
        if self.accelerator.is_main_process:
            import numpy as np
            all_acc_show = np.array(all_acc_show)

            metrics = {
                f"acc_eval": all_acc_show.mean(),
                f"eval_num": len(all_acc_show),
            }
            if 'medical' in self.task_name:
                all_acc_show_closed = np.array(all_acc_show_closed)
                metrics['acc_eval_closed'] = all_acc_show_closed.mean()
                metrics['eval_num_closed'] = len(all_acc_show_closed)

            # 4. 上报并返回
            self.log(metrics)
        self.model.train()
        return metrics

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch, our dataloader loads an *accumulated* batch
    # (i.e., `per_device_batch_size × gradient_accumulation_steps`). This allows us to generate completions
    # once per optimization step—rather than once per gradient accumulation step—which is significantly more efficient.
    # The only change from the original implementation is multiplying the batch size by `gradient_accumulation_steps`.
    # Thus, `_prepare_inputs` is called with the accumulated batch size, and it handles the splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification.As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.gradient_accumulation_steps,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Sampler:
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    # def _get_eval_sampler(self, eval_dataset) -> Sampler:
    #     # See _get_train_sampler for an explanation of the sampler.
    #     return RepeatSampler(
    #         data_source=eval_dataset,
    #         mini_repeat_count=self.num_generations,
    #         seed=self.args.seed,
    #     )
    def _get_eval_sampler(self, eval_dataset):
        # eval_dataset 是一个 map-style Dataset（非 IterableDataset）
        return DistributedSampler(
            dataset=eval_dataset,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            shuffle=False,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    @profiling_decorator
    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, pixel_attention_mask, logits_to_keep, batch_size=None) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]
            pixel_values_batch = pixel_values[i : i + batch_size]
            pixel_attention_mask_batch = pixel_attention_mask[i : i + batch_size]
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch, pixel_values=pixel_values_batch,
                       pixel_attention_mask=pixel_attention_mask_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    @profiling_decorator
    def _prepare_inputs(
        self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the accumulated local batch (Per-GPU batch size × Gradient accumulation steps)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire accumulated batch and splits it into smaller batches
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every gradient_accumulation_steps * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.gradient_accumulation_steps * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                accumulated_local_batch = self._generate_and_score_completions(accumulated_local_batch)
                self._buffered_inputs = split_tensor_dict(
                    accumulated_local_batch, self.args.gradient_accumulation_steps
                )
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, there is neither gradient accumulation, nor multiple iterations
            inputs = self._generate_and_score_completions(accumulated_local_batch)
        return inputs

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        dt_train_list = []
        dt_generate_list = []
        prompts = []
        question_wo_prompt_list = []
        images = []
        end_gt_list = []
        start_gt_list = []
        for x in inputs:
            question = x["prompt"]
            answer = x["answer"]
            image = x["image"]
            question_wo_prompt = x["only_q"]

            dt_train = {"question": question_wo_prompt, "image": image, "answer": answer}
            dt_generate = {"question": question, "image": image, "answer": None}

            question_wo_prompt_list.append(question_wo_prompt)

            end_prompt = self.end_template % answer if self.end_template else answer
            # call

            start_prompt = self.start_template % answer if self.start_template else answer

            # print(knowledge_with_end)
            dt_train_list.append(dt_train)
            dt_generate_list.append(dt_generate)
            # questions.append(dt_generate_q)
            prompts.append(question)
            images.append(image)
            end_gt_list.append(end_prompt)
            start_gt_list.append(start_prompt)

        if self.is_sft:
            # if self.sft_start >= 0.5:
            # sft train
            dt_train_dt = self.processing_func(dt_train_list)

            # 每隔num取一个
            for k in dt_train_dt:
                dt_train_dt[k] = dt_train_dt[k][::self.num_generations]

            return dt_train_dt

        dt_generate_dt = self.processing_func(dt_generate_list)
        # print(self.accelerator.device, "=========================", prompts)
        # dt_answer_dt = self.processing_func(knowledge_with_end_list)
        prompt_inputs_generate = super(MyGRPOTrainer, self)._prepare_inputs(dt_generate_dt)
        del prompt_inputs_generate["labels"]
        prompt_ids = prompt_inputs_generate["input_ids"]
        prompt_mask = prompt_inputs_generate["attention_mask"]
        pixel_values = prompt_inputs_generate["pixel_values"]
        pixel_attention_mask = prompt_inputs_generate["pixel_attention_mask"]

        # Regular generation path
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ):
                prompt_completion_ids = unwrapped_model.generate(**prompt_inputs_generate, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()


        ######################## my ##########################
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device)
        images_path = [image if isinstance(image, str) else image.filename for image in images]
        # 区分答案与reason
        start_list = []
        context_list = []
        end_list = []
        start_ans_list = []
        end_ans_list = []

        for completion in completions:
            start, context, end = split_initial_context_final(completion)
            start_list.append(start)
            context_list.append(context)
            end_list.append(end)
            start_ans = start.lower().split("the initial answer is ")[-1]
            end_ans = end.lower().split("the final answer is ")[-1]
            start_ans_list.append(start_ans)
            end_ans_list.append(end_ans)

        # start_acc_reward = self.reward_funcs[0]
        end_acc_reward_func = self.reward_funcs[-1]
        len_reward_func = self.reward_funcs[1]
        format_reward_func = self.reward_funcs[0]

        knowledge_list = [x["knowledge"] for x in inputs]

        standard_answers = [x["answer"] for x in inputs]
        tp_list = [x['tp'] for x in inputs] if 'tp' in inputs[0] else None

        data_dt = {'prompt': question_wo_prompt_list, 'hints': knowledge_list,
                   'image': images_path, 'standard_answer': standard_answers, 'task': [self.task_name]*len(images_path)}
        if tp_list is not None:
            data_dt['tp'] = tp_list
        gpu_id = self.accelerator.device.index

        data_dt['response'] = end_ans_list
        end_acc_rewards = torch.tensor(end_acc_reward_func(data_dt, gpu_id), dtype=torch.float32, device=device)
        data_dt['response'] = completions
        format_rewards = torch.tensor(format_reward_func(data_dt, gpu_id), dtype=torch.float32, device=device)
        data_dt['response'] = completions
        len_rewards = torch.tensor(len_reward_func(data_dt, gpu_id), dtype=torch.float32, device=device)
        rewards_per_func[:, 0] = format_rewards.clone()
        rewards_per_func[:, 1] = len_rewards.clone()
        rewards_per_func[:, -1] = end_acc_rewards.clone()
        len_rewards[end_acc_rewards == 0] = 0
        end_acc_rewards = end_acc_rewards * 2

        rewards_per_func_real = rewards_per_func.clone()
        rewards_per_func_real[:, 1] = len_rewards
        rewards_per_func_real[:, -1] = end_acc_rewards

        rewards_per_func_real = gather(rewards_per_func_real)
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func_real * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        advantages = advantages.reshape(-1, 1)
        end_acc_rewards = end_acc_rewards.view(-1, self.num_generations)
        format_rewards = format_rewards.view(-1, self.num_generations)

        has_format_correct = (end_acc_rewards > 0.5).sum(1)
        # has_format_correct = ((end_acc_rewards > 0.5) | (format_rewards > 0.5)).sum(1)
        has_correct = (end_acc_rewards > 0.5).sum(1)
        # has_correct = ((end_acc_rewards > 0.5) | (format_rewards > 0.5)).sum(1)
        format_rewards = format_rewards.view(-1)

        check_list = []
        for i in range(len(context_list)):
            batch_id = i // self.num_generations
            check_list.append((has_format_correct[batch_id] == 0) & (i % self.num_generations == 0))
        sents = more_express(question_wo_prompt_list, standard_answers, check_list, images_path, knowledge_list,
                             gpu_id=self.accelerator.device.index, tasks=[self.task_name] * len(images_path))
        for i in range(len(sents)):
            sent = sents[i]
            end = end_gt_list[i]
            if sent != "" and sent[-1] == ",":
                end = end[0].lower() + end[1:]
            end = sent + " " + end
            end_gt_list[i] = end

        start_prompt_dt = self.processing_class.tokenizer(start_gt_list, return_tensors="pt", padding=True,
                                                          padding_side="right")
        end_prompt_dt = self.processing_class.tokenizer(end_gt_list, return_tensors="pt", padding=True,
                                                        padding_side="right")
        new_compeltion_ids = []
        new_attn_masks = []

        advantage_list = []
        for i in range(len(context_list)):
            add_ct = True
            reward = advantages[i].squeeze()
            end = end_gt_list[i].strip()
            # context = context_list[i].strip()
            # new_completion = context + ' ' + end

            context = ""
            new_completion = " " + end

            start_id = self.processing_class(text=context, return_tensors="pt").input_ids[0]
            all_dt = self.processing_class(text=new_completion, return_tensors="pt")
            all_id = all_dt.input_ids[0]

            advantage_tensor = torch.zeros(len(all_id), device=device)
            format_check = format_rewards[i] > 0.5
            adv = 1
            advantage_tensor[:] = adv

            attn_mask = all_dt.attention_mask[0]
            new_compeltion_ids.append(all_id)
            advantage_list.append(advantage_tensor)
            new_attn_masks.append(attn_mask)

        completion_padded_ids = pad_sequence(new_compeltion_ids, batch_first=True,
                                             padding_value=self.processing_class.tokenizer.pad_token_id).long().to(
            device)
        completion_padded_advantages = pad_sequence(advantage_list, batch_first=True, padding_value=0).to(device)
        completion_padded_attn_masks = pad_sequence(new_attn_masks, batch_first=True, padding_value=0).to(device)

        final_completion_id_list = []
        final_completion_mask_list = []
        final_advantange_list = []

        for i in range(len(completion_padded_ids)):
            batch_id = i // self.num_generations
            if has_format_correct[batch_id] == 0:
                fake = torch.zeros(self.num_generations)
                fake[0] = 1
                fake = (fake - fake.mean()) / (fake.std() + 1e-4)
                # if random.random() < 0.5:
                if check_list[i]:  # 第一个修改为正确答案，其他的保留为错误的。
                    completion_id_ = torch.cat([completion_padded_ids[i], completion_ids[i][0:0]])
                    completion_mask_ = torch.cat([completion_padded_attn_masks[i], completion_mask[i][0:0]])
                    advantange_ = torch.cat([completion_padded_advantages[i], advantages[i][0:0]])
                    advantange_[:] = fake[0]
                else:
                    completion_id_ = torch.cat([completion_ids[i], completion_padded_advantages[i][0:0]])
                    completion_mask_ = torch.cat([completion_mask[i], completion_padded_attn_masks[i][0:0]])
                    advantange_ = torch.cat([advantages[i], completion_padded_advantages[i][0:0]])
                    advantange_ = advantange_.repeat_interleave(len(completion_id_))
                    advantange_[:] = fake[1] * 0

            else:
                completion_id_ = torch.cat([completion_ids[i], completion_padded_advantages[i][0:0]])
                completion_mask_ = torch.cat([completion_mask[i], completion_padded_attn_masks[i][0:0]])
                advantange_ = torch.cat([advantages[i], completion_padded_advantages[i][0:0]])
                # 如果advantange_是一个数字的话需要扩展维度
                advantange_ = advantange_.repeat_interleave(len(completion_id_))

            if has_correct[batch_id] == self.num_generations:  # 全部正确时停止优化
                advantange_[:] = 0

            final_completion_id_list.append(completion_id_)
            final_completion_mask_list.append(completion_mask_)
            final_advantange_list.append(advantange_)

        completion_ids = pad_sequence(final_completion_id_list, batch_first=True,
                                      padding_value=self.processing_class.tokenizer.pad_token_id).long()
        completion_mask = pad_sequence(final_completion_mask_list, batch_first=True, padding_value=0)
        completion_advantange = pad_sequence(final_advantange_list, batch_first=True, padding_value=0)
        completion_ids = completion_ids.to(device)
        completion_mask = completion_mask.to(device)
        completion_advantange = completion_advantange.to(device)
        input_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1).long()
        attention_completion_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        for s, a in enumerate(completion_advantange):
            if end_acc_rewards.view(-1)[s] > 0 and format_rewards.view(-1)[s] > 0 and a[0] < 0:
                print('no')

        if self.accelerator.device.index == 0:
            completion_id = completion_ids[0]
            completion_id_pos = completion_id[(completion_advantange[0] > 0) & (completion_mask[0] > 0)]
            completion_id_neg = completion_id[(completion_advantange[0] < 0) & (completion_mask[0] > 0)]

            show = self.processing_class.decode(completion_id_pos, skip_special_tokens=False)
            show_neg = self.processing_class.decode(completion_id_neg, skip_special_tokens=False)

            print(has_correct, completions[0], "\n=====POS====================\n", show,
                  "\n======NEG===================\n", show_neg)

            completion_id = completion_ids[1]
            completion_id_pos = completion_id[(completion_advantange[1] > 0) & (completion_mask[1] > 0)]
            completion_id_neg = completion_id[(completion_advantange[1] < 0) & (completion_mask[1] > 0)]

            show = self.processing_class.decode(completion_id_pos, skip_special_tokens=False)
            show_neg = self.processing_class.decode(completion_id_neg, skip_special_tokens=False)

            print(has_correct, completions[1], "\n=====POS====================\n", show,
                  "\n======NEG===================\n", show_neg)

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(self.model, input_completion_ids, attention_completion_mask, pixel_values,
                                          pixel_attention_mask, logits_to_keep, batch_size)
            else:
                old_per_token_logps = None

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        # completion_advantange: (batch_size, seq_len) 或 (batch_size, n)
        mask_pos = completion_advantange > 0  # 正优势位置
        row_min = completion_advantange.min(dim=1, keepdim=True).values.abs()  # (batch, 1)

        # 只对正优势加 abs(row_min)，其余位置设 0
        # completion_advantange = torch.where(
        #     mask_pos,
        #     completion_advantange + row_min,  # broadcasting 自动对齐到每一行
        #     torch.zeros_like(completion_advantange)
        # )
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "pixel_attention_mask": pixel_attention_mask,
            "pixel_values": pixel_values,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": completion_advantange,
            "old_per_token_logps": old_per_token_logps,
            "has_correct": has_correct,
        }

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = None

        # get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(unwrapped_model, input_ids, attention_mask, logits_to_keep)

        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"][:, 0],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs["old_per_token_logps"],
            ref_per_token_logps=ref_per_token_logps,
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        pixel_values, pixel_attention_mask = inputs["pixel_values"], inputs["pixel_attention_mask"]
        has_correct = inputs["has_correct"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, pixel_values,
                                          pixel_attention_mask, logits_to_keep)

        # sft_loss = -(per_token_logps * completion_mask).sum(-1) / completion_mask.sum(-1)
        advantages = inputs["advantages"][:, 0]
        # sft_loss = (sft_loss * (advantages > 0)).sum() * (has_correct == 0)
        # return sft_loss
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        # loss = (has_correct > 0) * loss + sft_loss
        # loss = (has_correct > 0) * loss
        # Log the metrics
        mode = "train" if self.model.training else "eval"

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self._textual_logs["rewards"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))