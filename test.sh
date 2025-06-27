#!/bin/bash

# 用法: ./run_eval.sh MODEL_NAME MODEL_PATH

MODEL_NAME=$1
MODEL_PATH=$2

# 配置数据集路径 + 是否是中文（true/false）
DATASETS=(
    "/mnt/general/liziheng/Qwen-Audio/datasets/CommonVoice/commonVoice_en_test_mini.jsonl,false"
    "/mnt/general/liziheng/Qwen-Audio/datasets/CommonVoice/commonVoice_zh_test_mini.jsonl,true"
    "/mnt/general/liziheng/Qwen-Audio/datasets/Fleurs/fleurs_en_test.jsonl,false"
    "/mnt/general/liziheng/Qwen-Audio/datasets/Fleurs/fleurs_zh_test.jsonl,true"
    "/mnt/general/liziheng/Qwen-Audio/datasets/WenetSpeech/wenet_test_meeting_mini.jsonl,true"
    "/mnt/general/liziheng/Qwen-Audio/datasets/WenetSpeech/wenet_test_net_mini.jsonl,true"
    "/mnt/general/liziheng/Qwen-Audio/datasets/LibriSpeech/LibriSpeech_Test_mini.jsonl,false"
)

for ENTRY in "${DATASETS[@]}"
do
    IFS=',' read -r TEST_FILE IS_CHINESE <<< "$ENTRY"

    echo "Running evaluation on dataset: $TEST_FILE (Chinese: $IS_CHINESE)"

    CMD="python ASR-eval.py --model_path \"$MODEL_PATH\" --test_data \"$TEST_FILE\" --model_name \"$MODEL_NAME\""
    
    if [ "$IS_CHINESE" == "true" ]; then
        CMD="$CMD --is_Chinese"
    fi

    eval $CMD
done
