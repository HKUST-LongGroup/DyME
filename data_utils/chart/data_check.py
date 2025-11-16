import json

data = json.load(open('/path/to/chartqa_output/json/train.json'))

for d in data:
    if 'answer' not in d:
        print(d)