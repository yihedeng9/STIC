#!/bin/bash
export HF_HOME="/data1/yihedeng"

python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-v1.6-mistral-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.6-mistral-7b-graph.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --load-peft /data1/yihedeng/checkpoints/llava_graph_qa_curriculum 

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.6-mistral-7b-graph.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.6-mistral-7b-graph_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.6-mistral-7b-graph_result.json
