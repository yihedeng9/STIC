#!/bin/bash
# export HF_HOME="/data1/yihedeng"
ANSWER="llava-v1.6-7b-stic"

python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.6-mistral-7b \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$ANSWER.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --load-peft ydeng9/llava-v1.6-mistral-7b-STIC

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$ANSWER.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$ANSWER.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$ANSWER.jsonl
