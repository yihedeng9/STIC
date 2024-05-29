import json
import os
import argparse

# initialize parser 
parser = argparse.ArgumentParser()
# add arguments
parser.add_argument("--input-file", type=str, default="pref_data.json")

# parse arguments
args = parser.parse_args()

# read input file
with open(args.input_file, "r") as f:
    data = json.load(f)

# print(data[0]) # {'image', 'chosen'}

# convert data to sft format
# sft format is a list of dictionaries, with keys ['conversations', 'id', 'image']

sft_data = []
for line in data:
    sft_line = {}
    sft_line['image'] = line['image']#.replace('COCO_train2014_', '')
    sft_line['id'] = sft_line['image'].replace('COCO_train2014_', '').replace('.jpg', '')
    sft_line['conversations'] = [{'from': 'human', 'value': line['chosen'][0]['content']},
                                 {'from': 'gpt', 'value': line['chosen'][1]['content']}]
    sft_data.append(sft_line)

# write sft data to file
output_file = args.input_file.replace('.json', '_sft.json')
with open(output_file, "w") as f:
    json.dump(sft_data, f)