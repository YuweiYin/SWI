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
    EVAL_NUM=${val}
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

if [[ -z ${EVAL_NUM} ]]; then
  EVAL_NUM="100"
fi

MODEL_NAME="${MODEL//[\/]/--}"
SEED=42

echo -e "TASK: ${TASK}"
echo -e "MODEL: ${MODEL}"
echo -e "MODEL_NAME: ${MODEL_NAME}"
echo -e "BSZ: ${BSZ}"
echo -e "EVAL_TASKS: ${EVAL_TASKS}"
echo -e "GEN_TEMP: ${GEN_TEMP}"
echo -e "EVAL_NUM: ${EVAL_NUM}"
echo -e "RANDOM SEED: ${SEED}"

CACHE_DIR=$2
PROJECT_DIR=$3
OUTPUT_DIR=$4
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface"
fi
if [[ -z ${PROJECT_DIR} ]]; then
  PROJECT_DIR="${HOME}/projects/SWI"
fi
if [[ -z ${OUTPUT_DIR} ]]; then
  # Make sure "${OUTPUT_DIR}/${EVAL_TASKS}" exists and contains the target task/dataset to evaluate
  OUTPUT_DIR="${PROJECT_DIR}/results/swi_gen_eval-temp_${GEN_TEMP}"  # Baseline
  #OUTPUT_DIR="${PROJECT_DIR}/results/swi_gen_eval-temp_${GEN_TEMP}--swi"  # SWI
fi
echo -e "CACHE_DIR: ${CACHE_DIR}"
echo -e "PROJECT_DIR: ${PROJECT_DIR}"
echo -e "OUTPUT_DIR: ${OUTPUT_DIR}"

if [[ ${EVAL_TASKS} == "SUM_ALL" ]]; then
  EVAL_TASK_NAME="cnn_dailymail,xsum,xlsum,dialogsum,wiki_lingua"
else
  EVAL_TASK_NAME="${EVAL_TASKS}"
fi

OPENAI_MODEL="gpt-4o-mini"
OPENAI_API_KEY=$5
if [[ -z ${OPENAI_API_KEY} ]]; then
  OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
fi

echo -e "\n\n >>> python3 run_eval_prf.py --eval_task_name ${EVAL_TASK_NAME} --hf_id ${MODEL} --results_dir ${OUTPUT_DIR}"
python3 run_eval_prf.py \
  --task "${TASK}" \
  --eval_task_name "${EVAL_TASK_NAME}" \
  --hf_id "${MODEL}" \
  --cache_dir "${CACHE_DIR}" \
  --project_dir "${PROJECT_DIR}" \
  --results_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --bsz "${BSZ}" \
  --openai_model "${OPENAI_MODEL}" \
  --openai_api_key "${OPENAI_API_KEY}" \
  --eval_num "${EVAL_NUM}" \
  --overwrite \
  --verbose
