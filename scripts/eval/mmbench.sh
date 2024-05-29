#!/bin/bash
export HF_HOME="/data1/yihedeng"
SPLIT="mmbench_dev_20230712"

#python -m llava.eval.model_vqa_mmbench \
#    --model-path liuhaotian/llava-v1.6-mistral-7b \
#    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
#    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-v1.6-7b-graph.jsonl \
#    --single-pred-prompt \
#    --temperature 0 \
#    --conv-mode vicuna_v1 \
#    --load-peft /data1/yihedeng/checkpoints/llava_graph_qa_curriculum 

#mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT 

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.6-7b-graph 
