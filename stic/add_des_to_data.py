import json

des = []
# read all jsonl files into a list data
#for i in range(1,5):
#    with open(f"image_description_part{i}.jsonl", 'r') as json_file:
#        json_list = list(json_file)

#    for json_str in json_list:
#        des.append(json.loads(json_str))

with open(f"image_description.jsonl", 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    des.append(json.loads(json_str))

with open("data/mixed_5k.json", "r") as f:
    sft_data = json.load(f)

for i in range(len(data)):
    cap = data[i]["description"]

    if "<image>\n" in sft_data[i]["conversations"][0]["value"]:
        sft_data[i]["conversations"][0]["value"] = sft_data[i]["conversations"][0]["value"].replace("<image>\n","<image>\nImage description:\n"+cap+"\n\n")
    else:
        sft_data[i]["conversations"][0]["value"] = "Image description:\n" + cap + "\n\n" + sft_data[i]["conversations"][0]["value"]

print(sft_data[0])
with open("data/mixed_5k_description.json", "w") as f:
    json.dump(sft_data, f, indent=4)
