import argparse
import torch
import torchvision.transforms as T
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import random
import json

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.eval.run_llava import eval_model

import requests
from PIL import Image
from io import BytesIO
import re
import os

from tqdm import tqdm 
random.seed(42)

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file, image_corruption=False):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    if image_corruption:
        if random.random() > 0.5:
            image = T.Resize(size=20)(image)
        else:
            jitter = T.ColorJitter(brightness=.5, hue=.3)
            image = jitter(image)
    return image


def load_images(image_files, image_corruption=False):
    out = []
    for image_file in image_files:
        image = load_image(image_file, image_corruption)
        out.append(image)
    return out


def eval_model(args, image_corruption=False):

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = args.image_file 
    images = load_images([image_files], image_corruption)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default="/data/user/MSCOCO/train2014")
    parser.add_argument("--save-dir", type=str, default="pref_data_mscoco.jsonl")
    parser.add_argument("--image-file", type=str, default="/data/user/MSCOCO/val2014/COCO_val2014_000000033958.jpg")
    parser.add_argument("--query", type=str, default="Describe the image.")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    prompt_list = ["Illustrate the details of the picture.",
                   "Summarize the visual content presented.",
                   "Explain what is depicted in the photograph.",
                   "Outline the key elements captured in the image.",
                   "Detail the composition and subjects within the frame.",
                   "Convey the atmosphere and mood represented in the snapshot.",
                   "Interpret the scene shown in the image.",
                   "Identify and describe the main focal points in the visual."]
    
    full_prompt = """Please provide a detailed description of the image, focusing on the following. 
    Identify the main subjects (people, animals, objects) in the image and describe what they are doing.
    Describe the setting of the image. Is it indoors or outdoors? What kind of environment or location does it depict? 
    What mood does the image convey? Are there any specific elements (such as lighting, weather, expressions) that contribute to this atmosphere? 
    Describe the dominant colors and the overall composition. How do these elements affect the image's impact?
    Point out any details or symbols that might be relevant to understanding the image's meaning or context.
    If applicable, provide interpretations of what the image might represent or communicate."""
    
    hallu_prompt_list = ["Describe the image with imaginative objects that may exist in the scene.",
                         "Enrich the description by adding hypothetical objects or characters that could be part of the scene.",
                         "Suggest and detail practical items or people that could logically inhabit the image's setting.",
                         "Incorporate elements that, though absent, would seamlessly fit into the context of the picture.",
                         "Imagine and describe additional everyday objects or activities taking place just out of frame.",
                         "Augment the scene with details of potential events or items that are plausible.",
                         "Conceive of and detail natural elements, such as weather or animals, that could realistically enter the scene. Make the description affirmative.",
                         "Invent and incorporate details of practical tools, vehicles, or gadgets that could be expected in a similar scenario."]


    directory = args.image_dir
    coco = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    random.shuffle(coco)

    for i in tqdm(range(len(coco))):
        image_name = coco[i]
        args.image_file = f"{args.image_dir}/{image_name}"
        args.query = full_prompt
        image_corruption = False

        preferred_output = eval_model(args, image_corruption)
        
        hallu_prompt = ""
        prompt = random.choice(prompt_list)

        # random sample a number between 0 and 1
        if random.random() > 0.5:
            hallu_prompt = random.choice(hallu_prompt_list)
            args.query = hallu_prompt
            image_corruption = False
            corrupted_output = eval_model(args)
        else:
            args.query = prompt
            image_corruption = True
            corrupted_output = eval_model(args)

        d = {"image": image_name, 
             "image_corruption": image_corruption,
             "hallu_prompt": hallu_prompt,
             "chosen": [{"role":"user","content":prompt},{"role":"assistant","content":preferred_output}],
             "rejected": [{"role":"user","content":prompt},{"role":"assistant","content":corrupted_output}]}
        
        with open(args.save_dir,"a") as f:
            f.write(json.dumps(d))
            f.write("\n")
