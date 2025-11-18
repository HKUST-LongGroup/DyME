import os.path

from PIL import Image as PILImage

from data_utils.aokvqa.data_collector import prepare_world_rl_data, prepare_world_sft_data, prepare_world_dyme_data
from data_utils.chart.data_collector import prepare_chart_rl_data, prepare_chart_sft_data
from data_utils.lm_math.data_collector import prepare_math_lm_rl_data

prompt_ic = """
Based on the provided sentence <C>, extract all the visual elements. Organize them into a structured format that can be directly converted into a Python list. 

Note: visual elements are all the things that can be seen in a sentence - tangible, perceivable items, places, people, colors, shapes, movements, etc.

Here are some examples:
<C>: A small black cat is sitting on a wooden table under the bright sunlight.
Output: [
    {"object": "cat", "attributes": ["small", "black"], "action": "sitting"},
    {"object": "table", "attributes": ["wooden"]},
    {"environment": "sunlight", "attributes": ["bright"]}
    {"description": "The scene is illuminated by bright sunlight..."}
]

<C>: "Year | Favorable | Unfavorable \n 2011 | 0 | 3.1 \n 2012 | 56 | 38.0 \n 2013 | 0 | 0.0 \n 2014 | 51 | 48.0 \n 2015 | 0 | 53.0"
Output: [
    {"Year": 2011, "Favorable": 0, "Unfavorable": 3.1},
    {"Year": 2012, "Favorable": 56, "Unfavorable": 38.0},
    {"Year": 2013, "Favorable": 0, "Unfavorable": 0.0},
    {"Year": 2014, "Favorable": 51, "Unfavorable": 48.0},
    {"Year": 2015, "Favorable": 0, "Unfavorable": 53.0}
]

<C>: The old castle stands on a rocky hill surrounded by mist.
Output: [
    {"object": "castle", "attributes": ["old"], "position": "stands"},
    {"object": "hill", "attributes": ["rocky"]},
    {"environment": "mist"}
    {"description": "The castle is situated on a rocky hill enveloped in mist..."}
]

Now, following the examples above, please extract the visual element from the sentence without providing any explanation or comments.

<C>: %s
Your Output:
"""

def collate_fn(examples, processor, label_id=151646):

    texts = []
    images = []
    for example in examples:
      image = example["image"]
      if isinstance(image, str):
        if not os.path.exists(image):
            image = image.replace('/chartqa_output/', '/apdcephfs_nj4/share_300377003/realzliu/data/chartqa_output/')
        image = PILImage.open(image)
      if image.mode != 'RGB':
        image = image.convert('RGB')
      question = example["prompt"]
      answer = example.get("answer", None)
      if answer is not None:
          messages = [
              {
                  "role": "user",
                  "content": [
                      # {"type": "text", "text": "Answer briefly."},
                      {"type": "image"},
                      {"type": "text", "text": question}
                  ]
              },
              {
                  "role": "assistant",
                  "content": [
                      {"type": "text", "text": answer}
                  ]
              }
          ]
          text = processor.apply_chat_template(messages, add_generation_prompt=False)
          texts.append(text.strip())
      else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(text.strip())

      images.append(image)
    # print(texts)
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    if label_id is not None:
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == label_id] = -100
        batch["labels"] = labels

    return batch

def collate_fn_woI(examples, processor, label_id=151646):

    texts = []
    images = []
    for example in examples:
      question = example["prompt"]
      answer = example.get("answer", None)
      if answer is not None:
          # --- FIX 1: "content" is now a simple string ---
          messages = [
              {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
              {"role": "user", "content": question},
              {"role": "assistant", "content": answer}
          ]
          text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
          texts.append(text.strip())
      else:
          # --- FIX 1: "content" is now a simple string ---
          messages = [
              {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
              {"role": "user", "content": question}
          ]
          text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
          texts.append(text.strip())

    # print(texts)
    batch = processor(text=texts, return_tensors="pt", padding=True)

    if label_id is not None:
        labels = batch["input_ids"].clone()
        labels[labels == processor.pad_token_id] = -100
        labels[labels == label_id] = -100
        batch["labels"] = labels

    return batch

def define_task_data_func(task, mode='rl'):
    if 'medical' in task:
        return None
    elif 'chart' in task:
        if mode == 'rl':
            return prepare_chart_rl_data
        return prepare_chart_sft_data
    elif 'math' == task:
        return None
    elif 'math_lm' in task:
        return prepare_math_lm_rl_data
    elif 'world' in task:
        if mode == 'rl':
            return prepare_world_rl_data
        elif mode == 'sft':
            return prepare_world_sft_data
        return prepare_world_dyme_data
    else:
        return None