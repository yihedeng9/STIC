import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Path to jsonl file")

args = parser.parse_args()

with open(args.input, 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    data.append(json.loads(json_str))

with open(args.input.replace('.jsonl', '.json'), 'w') as json_file:
    json.dump(data, json_file
