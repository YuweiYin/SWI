#!/bin/bash

CACHE_DIR=$1
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/projects/def-carenini/yuweiyin/.cache/huggingface/"
fi

HF_TOKEN=$2
if [[ -z ${HF_TOKEN} ]]; then
  HF_TOKEN="YOUR_HF_TOKEN"  # https://huggingface.co/settings/tokens
fi

# Mathematical Reasoning
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "openai/gsm8k" --subset "main"  # GSM8K
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "madrylab/gsm8k-platinum"  # GSM8K-Platinum
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "HuggingFaceH4/MATH-500"  # MATH500

# Text Summarization
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "abisee/cnn_dailymail" --subset "3.0.0"  # CNN / DailyMail
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "EdinburghNLP/xsum"  # XSum
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "GEM/xlsum" --subset "english"  # XL-Sum
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "knkarthick/dialogsum"  # DialogSum
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "GEM/wiki_lingua"  # Wikilingua

# Multi-task Multiple-choice Question Answering
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "TIGER-Lab/MMLU-Pro"  # MMLU-Pro

# Multiple-choice QA - Big-Bench Hard (BBH)
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "word_sorting"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "web_of_lies"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "tracking_shuffled_objects_three_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "tracking_shuffled_objects_seven_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "tracking_shuffled_objects_five_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "temporal_sequences"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "sports_understanding"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "snarks"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "salient_translation_error_detection"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "ruin_names"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "reasoning_about_colored_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "penguins_in_a_table"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "object_counting"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "navigate"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "multistep_arithmetic_two"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "movie_recommendation"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "logical_deduction_three_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "logical_deduction_seven_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "logical_deduction_five_objects"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "hyperbaton"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "geometric_shapes"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "formal_fallacies"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "dyck_languages"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "disambiguation_qa"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "date_understanding"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "causal_judgement"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "lukaemon/bbh" --subset "boolean_expressions"

# Multiple-choice QA - MMLU
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "formal_logic"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "philosophy"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_world_history"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "international_law"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "jurisprudence"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "world_religions"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "moral_disputes"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_european_history"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "logical_fallacies"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_us_history"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "moral_scenarios"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "professional_law"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "prehistory"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "us_foreign_policy"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "security_studies"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "econometrics"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_microeconomics"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "sociology"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_geography"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_psychology"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "professional_psychology"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_macroeconomics"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_government_and_politics"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "public_relations"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "human_sexuality"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "miscellaneous"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "medical_genetics"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "management"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "virology"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "nutrition"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "global_facts"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "marketing"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "college_medicine"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "clinical_knowledge"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "professional_accounting"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "professional_medicine"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "human_aging"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "business_ethics"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "college_physics"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "elementary_mathematics"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "machine_learning"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_statistics"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "electrical_engineering"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "college_computer_science"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "anatomy"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_physics"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_computer_science"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "computer_security"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "conceptual_physics"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "college_mathematics"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "astronomy"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_mathematics"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "college_chemistry"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "abstract_algebra"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_chemistry"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "college_biology"
python3 utils/download_hf_dataset.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code \
  --hf_id "hails/mmlu_no_train" --subset "high_school_biology"
