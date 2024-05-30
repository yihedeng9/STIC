#!/bin/bash
export HF_HOME="/data1/yihedeng"
SPLIT="mmbench_dev_20230712"
ANSWER="llava-v1.6-7b-stic-final"

python -m llava.eval.model_vqa_mmbench \
   --model-path liuhaotian/llava-v1.6-mistral-7b \
   --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
   --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$ANSWER.jsonl \
   --single-pred-prompt \
   --temperature 0 \
   --conv-mode vicuna_v1 \
   --load-peft STIC-LVLM/llava-v1.6-mistral-7b-STIC

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT 

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $ANSWER
