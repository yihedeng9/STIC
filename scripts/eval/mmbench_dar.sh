#!/bin/bash
export HF_HOME="/data1/yihedeng"
SPLIT="mmbench_dev_20230712"
ANSWER="llava-v1.6-7b-stic-dar"

python -m llava.eval.model_vqa_mmbench_dar \
   --model-path liuhaotian/llava-v1.6-mistral-7b \
   --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
   --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$ANSWER.jsonl \
   --single-pred-prompt \
   --temperature 0 \
   --conv-mode vicuna_v1 \
   --load-peft /data1/yihedeng/checkpoints/llava_coco_test

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT 

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $ANSWER
