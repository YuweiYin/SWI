#!/bin/bash

CACHE_DIR=$1
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface/"
fi

HF_TOKEN=$2
if [[ -z ${HF_TOKEN} ]]; then
  HF_TOKEN="YOUR_HF_TOKEN"  # https://huggingface.co/settings/tokens
fi

python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code --verbose \
  --hf_id "meta-llama/Llama-3.1-8B-Instruct"

python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code --verbose \
  --hf_id "meta-llama/Llama-3.2-3B-Instruct"
python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code --verbose \
  --hf_id "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code --verbose \
  --hf_id "FacebookAI/roberta-large"  # For BERTScore and Sentence Transformer metrics
python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code --verbose \
  --hf_id "FacebookAI/xlm-roberta-large"  # For BERTScore and Sentence Transformer metrics
