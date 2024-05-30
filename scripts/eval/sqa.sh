#!/bin/bash
export HF_HOME="/data1/yihedeng"
ANSWER="llava-v1.6-7b-stic"

python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-v1.6-mistral-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$ANSWER.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --load-peft STIC-LVLM/llava-v1.6-mistral-7b-STIC

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$ANSWER.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$ANSWER_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$ANSWER_result.json
