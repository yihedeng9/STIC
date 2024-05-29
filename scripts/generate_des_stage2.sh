export HF_HOME="/data1/yihedeng"

CUDA_VISIBLE_DEVICES=1 python ./stic/generate_des_stage2.py \
    --model-path liuhaotian/llava-v1.6-mistral-7b \
    --image-dir /data1/yihedeng/image_data \
    --save-dir image_description.jsonl \
    --adapter-path /data1/yihedeng/checkpoints/llava_coco_test 