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
    EVAL_METRICS=${val}
  elif [[ "${idx}" == "6" ]]; then
    GEN_TEMP=${val}
  elif [[ "${idx}" == "7" ]]; then
    LLM_JUDGE_TYPE=${val}
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

if [[ -z ${EVAL_METRICS} ]]; then
  EVAL_METRICS="ALL"
fi

if [[ -z ${GEN_TEMP} ]]; then
  GEN_TEMP="0.0"
fi

if [[ -z ${LLM_JUDGE_TYPE} ]]; then
  LLM_JUDGE_TYPE="1"
fi

MODEL_NAME="${MODEL//[\/]/_}"
SEED=42

echo -e "TASK: ${TASK}"
echo -e "MODEL: ${MODEL}"
echo -e "MODEL_NAME: ${MODEL_NAME}"
echo -e "BSZ: ${BSZ}"
echo -e "EVAL_TASKS: ${EVAL_TASKS}"
echo -e "EVAL_METRICS: ${EVAL_METRICS}"
echo -e "LLM_JUDGE_TYPE: ${LLM_JUDGE_TYPE}"
echo -e "GEN_TEMP: ${GEN_TEMP}"
echo -e "RANDOM SEED: ${SEED}"

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
  # Make sure "${OUTPUT_DIR}/${EVAL_TASKS}" exists and contains the target task/dataset to evaluate
  OUTPUT_DIR="${PROJECT_DIR}/results/swi_gen_eval-temp_${GEN_TEMP}"  # Baseline
  #OUTPUT_DIR="${PROJECT_DIR}/results/swi_gen_eval-temp_${GEN_TEMP}--swi"  # SWI
fi
echo -e "CACHE_DIR: ${CACHE_DIR}"
echo -e "PROJECT_DIR: ${PROJECT_DIR}"
echo -e "OUTPUT_DIR: ${OUTPUT_DIR}"

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

echo -e "\n\n >>> python3 run_eval_lm.py --eval_task_name ${EVAL_TASK_NAME} --hf_id ${MODEL} --output_dir ${OUTPUT_DIR}"
python3 run_eval_lm.py \
  --task "${TASK}" \
  --eval_task_name "${EVAL_TASK_NAME}" \
  --eval_metric_name "${EVAL_METRICS}" \
  --llm_judge_type "${LLM_JUDGE_TYPE}" \
  --hf_id "${MODEL}" \
  --cache_dir "${CACHE_DIR}" \
  --project_dir "${PROJECT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --bsz "${BSZ}" \
  --overwrite \
  --verbose
