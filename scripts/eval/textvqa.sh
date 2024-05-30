#!/bin/bash
ANSWER="llava-v1.6-7b-stic"

python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-mistral-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$ANSWER.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --load-peft STIC-LVLM/llava-v1.5-mistral-7b-STIC

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$ANSWER.jsonl
