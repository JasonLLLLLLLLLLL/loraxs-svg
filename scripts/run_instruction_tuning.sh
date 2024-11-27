#BASE_MODEL="google/gemma-7b"
# BASE_MODEL="mistralai/Mistral-7B-v0.1"
# BASE_MODEL= "/home/lz/SSD-GreatWall/models/AI-ModelScope/Mistral-7B-v0.1/"
# python -m utils.merge_adapter_to_base_model --base_mode "/home/lz/SSD-GreatWall/models/AI-ModelScope/Mistral-7B-v0.1/" --adapter "$FT_PATH" --output_path "$MERGED_PATH"

# python instruction_tuning_eval/gsm8k_eval.py --model "$MERGED_PATH"
# python instruction_tuning_eval/MATH_eval.py --model "$MERGED_PATH"
OUTPUT="output_32"
MERGED_PATH="output_merged_32"
LORA_RANK=128

export CUDA_VISIBLE_DEVICES=1


python main_instruction_tuning.py \
    --model_name_or_path "/home/liuzhe/new-files/AI-ModelScope/Mistral-7B-v0___1" \
    --output_dir $OUTPUT \
    --lora_r $LORA_RANK \
    --data_path /home/liuzhe/new-files/LoRA-XS/utils/dataset-1024-everypath-10-26.json \
    --dataset_split "train"\
    --dataset_field caption code \
    --num_train_epochs 168 \
    --per_device_train_batch_size 1\
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 16 \
    --save_total_limit 20 \
    --learning_rate 4e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 False \
    --tf32 False \
    --fp16 True \

FT_PATH=$(find $OUTPUT -type d -path "*/ft" | grep "/home/liuzhe/new-files/AI-ModelScope/Mistral-7B-v0___1" | grep rank_$LORA_RANK)
#python -m utils.merge_adapter_to_base_model --base_mode "/home/liuzhe/new-files/AI-ModelScope/Mistral-7B-v0___1" --adapter "$FT_PATH" --output_path "$MERGED_PATH"
#python -m utils.merge_adapter_to_base_model --base_mode "/home/liuzhe/new-files/AI-ModelScope/Mistral-7B-v0___1" --adapter "/home/liuzhe/new-files/LoRA-XS/output_32/home/liuzhe/new-files/AI-ModelScope/Mistral-7B-v0___1/home/liuzhe/new-files/LoRA-XS/utils/dataset-1024-everypath-10-26.json_split_train/LoRA_init_svd_rank_128_lr_0.004_seed_42/output_2024-11-22 09:15:04.830781/checkpoint-37728-0.32" --output_path "/home/liuzhe/new-files/LoRA-XS/output_merged_TSET"
