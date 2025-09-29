from PIL import Image as PILImage

from data_utils.chart.data_collector import deal_chart_rl_data

prompt_ic = """
Based on the provided sentence <C>, extract all the visual elements. Organize them into a structured format that can be directly converted into a Python list. 

Note: visual elements are all the things that can be seen in a sentence - tangible, perceivable items, places, people, colors, shapes, movements, etc.

Here are some examples:
C: A small black cat is sitting on a wooden table under the bright sunlight.
Output: [
    {"object": "cat", "attributes": ["small", "black"], "action": "sitting"},
    {"object": "table", "attributes": ["wooden"]},
    {"environment": "sunlight", "attributes": ["bright"]}
]

<C>: The old castle stands on a rocky hill surrounded by mist.
Output: [
    {"object": "castle", "attributes": ["old"], "position": "stands"},
    {"object": "hill", "attributes": ["rocky"]},
    {"environment": "mist"}
]

Now, following the examples above, please extract the visual element from the sentence without providing any explanation or comments.

<C>: %s
Your Output:
"""

def collate_fn(examples, processor, label_id=None):
    texts = []
    images = []
    for example in examples:
      image = example["image"]
      if isinstance(image, str):
        image = PILImage.open(image)
      if image.mode != 'RGB':
        image = image.convert('RGB')
      question = example["question"]
      answer = example["answer"]
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

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    if label_id is not None:
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == label_id] = -100
        batch["labels"] = labels

    return batch

def define_task_data_func(task):
    if 'medical' in task:
        return None
    elif 'chart' in task:
        return deal_chart_rl_data
    elif 'math' in task:
        return None
    else:
        return None