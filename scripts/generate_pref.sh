export HF_HOME="/data1/yihedeng"

CUDA_VISIBLE_DEVICES=5 python ./stic/generate_pref.py \
    --model-path liuhaotian/llava-v1.6-mistral-7b \
    --image-dir /data1/yihedeng/MSCOCO/train2014 \
    --save-dir pref_data_mscoco.jsonl \
