#!/bin/bash

PARAM=$1

IFS=";"  # Set ";" as the delimiter
read -ra PARAM_ARRAY <<< "${PARAM}"

#echo ${#PARAM_ARRAY[@]}
idx=0
for val in "${PARAM_ARRAY[@]}";
do
  idx=$(( $((idx)) + 1 ))
  # echo -e ">>> idx = ${idx}; val = ${val}"
  if [[ "${idx}" == "1" ]]; then
    TASK=${val}
  elif [[ "${idx}" == "2" ]]; then
    MODEL=${val}
  elif [[ "${idx}" == "3" ]]; then
    BSZ=${val}
  elif [[ "${idx}" == "4" ]]; then
    EVAL_TASKS=${val}
  elif [[ "${idx}" == "5" ]]; then
    GEN_TEMP=${val}
  elif [[ "${idx}" == "6" ]]; then
    MAX_GEN_LEN=${val}
  fi
done

if [[ -z ${TASK} ]]; then
  TASK="1"
fi

if [[ -z ${BSZ} ]]; then
  BSZ="1"
fi

if [[ -z ${EVAL_TASKS} ]]; then
  echo -e "!!! Error EVAL_TASKS input: \"${EVAL_TASKS}\"\n"
  exit 1
fi

if [[ -z ${GEN_TEMP} ]]; then
  GEN_TEMP="0.0"
fi

if [[ -z ${MAX_GEN_LEN} ]]; then
  MAX_GEN_LEN="4096"
fi

MODEL_NAME="${MODEL//[\/]/_}"

echo -e "TASK: ${TASK}"
echo -e "MODEL: ${MODEL}"
echo -e "MODEL_NAME: ${MODEL_NAME}"
echo -e "BSZ: ${BSZ}"
echo -e "EVAL_TASKS: ${EVAL_TASKS}"
echo -e "GEN_TEMP: ${GEN_TEMP}"
echo -e "MAX_GEN_LEN: ${MAX_GEN_LEN}"

CACHE_DIR=$2
PROJECT_DIR=$3
OUTPUT_DIR=$4
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface"
fi
if [[ -z ${PROJECT_DIR} ]]; then
  PROJECT_DIR="${HOME}/projects/def-carenini/yuweiyin/projects/SWI"
fi
if [[ -z ${OUTPUT_DIR} ]]; then
  OUTPUT_DIR="${PROJECT_DIR}/results/swi_gen_eval-temp_${GEN_TEMP}"  # ${MODEL_NAME} is subdir
fi
echo -e "CACHE_DIR: ${CACHE_DIR}"
echo -e "PROJECT_DIR: ${PROJECT_DIR}"
echo -e "OUTPUT_DIR: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

SEED=42

if [[ ${EVAL_TASKS} == "ALL" ]]; then
  EVAL_TASK_NAME="gsm8k,gsm8k_platinum,math500,amc23,aime24,aime25,logiqa,commonsense_qa,social_iqa,openbookqa,ai2_arc,bbh,mmlu,mmlu_pro,cnn_dailymail,xsum,xlsum,samsum,dialogsum,wiki_lingua"
elif [[ ${EVAL_TASKS} == "MATH_ALL" ]]; then
  EVAL_TASK_NAME="gsm8k,gsm8k_platinum,math500,amc23,aime24,aime25"
elif [[ ${EVAL_TASKS} == "QA_ALL" ]]; then
  EVAL_TASK_NAME="logiqa,commonsense_qa,social_iqa,openbookqa,ai2_arc,bbh,mmlu,mmlu_pro"
elif [[ ${EVAL_TASKS} == "SUM_ALL" ]]; then
  EVAL_TASK_NAME="cnn_dailymail,xsum,xlsum,samsum,dialogsum,wiki_lingua"
else
  EVAL_TASK_NAME="${EVAL_TASKS}"
fi

echo -e "\n\n >>> python3 run_gen_lm.py --eval_task_name ${EVAL_TASK_NAME} --hf_id ${MODEL}"
python3 run_gen_lm.py \
  --task "${TASK}" \
  --eval_task_name "${EVAL_TASK_NAME}" \
  --hf_id "${MODEL}" \
  --cache_dir "${CACHE_DIR}" \
  --project_dir "${PROJECT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --bsz "${BSZ}" \
  --gen_temperature "${GEN_TEMP}" \
  --max_gen_len "${MAX_GEN_LEN}" \
  --verbose
