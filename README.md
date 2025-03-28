<div align="center">

# SWI: Speaking with Intent in Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2503.21544-b31b1b.svg)](https://arxiv.org/abs/2503.21544)

</div>

<details open><summary>Paper Abstract</summary>

* **SWI**: Speaking with Intent in Large Language Models
* **Authors**: [Yuwei Yin](https://www.yuweiyin.com/), [EunJeong Hwang](https://eujhwang.github.io/), and [Giuseppe Carenini](https://www.cs.ubc.ca/~carenini/)
* **Paper**: https://huggingface.co/papers/2503.21544

```text
Intent, typically clearly formulated and planned, functions as a cognitive framework for reasoning 
and problem-solving. This paper introduces the concept of Speaking with Intent (SWI) in large 
language models (LLMs), where the explicitly generated intent encapsulates the model's underlying 
intention and provides high-level planning to guide subsequent analysis and communication. By 
emulating deliberate and purposeful thoughts in the human mind, SWI is hypothesized to enhance the 
reasoning capabilities and generation quality of LLMs. Extensive experiments on mathematical reasoning 
benchmarks consistently demonstrate the superiority of Speaking with Intent over Baseline (i.e., 
generation without explicit intent). Moreover, SWI outperforms answer-trigger prompting methods 
Chain-of-Thought and Plan-and-Solve and maintains competitive performance with the strong method ARR 
(Analyzing, Retrieving, and Reasoning). Additionally, the effectiveness and generalizability of SWI 
are solidified on reasoning-intensive question answering (QA) and text summarization benchmarks, where 
SWI brings consistent improvement to the Baseline generation. In text summarization, SWI-generated 
summaries exhibit greater accuracy, conciseness, and factual correctness, with fewer hallucinations. 
Furthermore, human evaluations verify the coherence, effectiveness, and interpretability of the intent 
produced by SWI. This proof-of-concept study creates a novel avenue for enhancing LLMs' reasoning 
abilities with cognitive notions.
```

<img src="https://yuweiyin.com/files/img/2025-03-27-SWI.jpg" alt="SWI" width="800" height="auto">

</details>

## Development Environments

### GitHub Repos & Development Environment

<details><summary>Environment Setup</summary>

- **Python**: Python 3.10
- **GPU**: A single NVIDIA V100-32GB GPU
  - LLMs (8B parameters) `float16` inference mode only

```bash
git clone https://github.com/YuweiYin/SWI
cd SWI/
# Now, "/path/to/SWI/" is the project root directory

# https://docs.conda.io/projects/miniconda/en/latest/
conda create -n swi python=3.10 -y
conda activate swi

pip install -r requirements.txt -i https://pypi.org/simple/
pip install -e . -i https://pypi.org/simple/

# We can set the Hugging Face cache directory. The following is for the dataset cache.
export HF_HOME="/path/to/your/.cache/huggingface/datasets"  # Default: "${HOME}/.cache/huggingface/datasets/"
```

</details>

## Datasets and Models (Paper Section XX)

- Download the datasets and models beforehand if the computing nodes have no Internet access or HOME storage is limited.
- Please ensure `CACHE_DIR` and `HF_TOKEN` in the script are correct directories.

### Datasets

```bash
# https://huggingface.co/datasets
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
HF_TOKEN="YOUR_HF_TOKEN"  # https://huggingface.co/settings/tokens
bash run_download_datasets.sh "${CACHE_DIR}" "${HF_TOKEN}"  # Download data to "${CACHE_DIR}/datasets/"
```

### Models

```bash
# https://huggingface.co/models
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
HF_TOKEN="YOUR_HF_TOKEN"  # https://huggingface.co/settings/tokens
bash run_download_models.sh "${CACHE_DIR}" "${HF_TOKEN}"  # Download models to "${CACHE_DIR}/"
```

## Main Experiments on Mathematical Reasoning (Paper Section XX)

For each bash script, please ensure `CACHE_DIR` and `PROJECT_DIR` in the script are 
correct Hugging Face cache directory (default: `"~/.cache/huggingface/"`) and 
project root directory (`"/path/to/SWI/"`).

```bash
mkdir -p logs/  # where we save running logs
mkdir -p results/  # where we save experimental results
```

### Mathematical Reasoning (Paper Table XX)

<details><summary>Experimental Settings</summary>

- **Datasets**: GSM8K, GSM8K-Platinum (GSM8K-P), MATH500, AMC23, AIME24, and AIME25
  - [x] `gsm8k`: **GSM8K** - [Paper](https://arxiv.org/abs/2110.14168); [Dataset](https://huggingface.co/datasets/openai/gsm8k)
  - [x] `gsm8k_platinum`: **GSM8K-Platinum** (GSM8K-P) - [Paper](https://arxiv.org/abs/2502.03461); [Dataset](https://huggingface.co/datasets/madrylab/gsm8k-platinum)
  - [x] `math500`: **MATH500** - [MATH Paper](https://openreview.net/forum?id=7Bywt2mQsCe), [MATH Dataset](https://huggingface.co/datasets/EleutherAI/hendrycks_math); [MATH500 Paper](https://openreview.net/forum?id=v8L0pN6EOi), [MATH500 Dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
  - **Competition**-level Math Benchmarks:
    - [x] `amc23`: **AMC23** - [Source](https://artofproblemsolving.com/wiki/index.php/AMC_Problems_and_Solutions), [Dataset](https://huggingface.co/datasets/math-ai/amc23)
    - [x] `aime24`: **AIME24** - [Source](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions), [Dataset](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
    - [x] `aime25`: **AIME25** - [Source](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions), [Dataset](https://huggingface.co/datasets/math-ai/aime25)
- **Comparison**: (Zero-shot Settings)
  - [x] Baseline (LLM Generation w/o SWI)
  - [x] **SWI** (Ours): Require LLMs to speak with (their own) intent.
  - Previous Answer-Trigger Prompting Methods
    - [x] CoT (Zero-shot Chain-of-Thought Prompting) ([Paper](https://arxiv.org/abs/2205.11916))
    - [x] PS (Plan-and-Solve Prompting) ([Paper](https://aclanthology.org/2023.acl-long.147/))
    - [x] ARR (Analyzing, Retrieving, and Reasoning) ([Paper](https://arxiv.org/abs/2502.04689))
- **Models**:
  - [x] `meta-llama/Llama-3.1-8B-Instruct` ([Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))

</details>

<details><summary>Experiment Script</summary>

```bash
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
PROJECT_DIR="/path/to/SWI/"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
GEN_TEMP="0.0"
OUTPUT_DIR="${PROJECT_DIR}/results/swi_results-temp_${GEN_TEMP}"  # Baseline output directory

# [Reasoning & Answer Generation] **First**, freely generate answers with reasoning:
echo -e "\n\n >>> bash run_gen_lm.sh --hf_id ${MODEL} MATH_ALL [Baseline]"
bash run_gen_lm.sh "1;${MODEL};1;MATH_ALL;${GEN_TEMP};4096" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}"
echo -e "\n\n >>> bash run_gen_lm-swi.sh --hf_id ${MODEL} MATH_ALL [SWI]"
bash run_gen_lm-swi.sh "1;${MODEL};1;MATH_ALL;${GEN_TEMP};4096" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}--swi"
echo -e "\n\n >>> bash run_gen_lm-cot.sh --hf_id ${MODEL} MATH_ALL [CoT]"
bash run_gen_lm-cot.sh "1;${MODEL};1;MATH_ALL;${GEN_TEMP};4096" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}--cot"
echo -e "\n\n >>> bash run_gen_lm-ps.sh --hf_id ${MODEL} MATH_ALL [PS]"
bash run_gen_lm-ps.sh "1;${MODEL};1;MATH_ALL;${GEN_TEMP};4096" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}--ps"
echo -e "\n\n >>> bash run_gen_lm-arr.sh --hf_id ${MODEL} MATH_ALL [ARR]"
bash run_gen_lm-arr.sh "1;${MODEL};1;MATH_ALL;${GEN_TEMP};4096" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}--arr"

# [Answer Extraction & Evaluation] **Second**, extract the answers and evaluate them:
echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} MATH_ALL [Baseline]"
bash run_eval_lm.sh "1;${MODEL};1;MATH_ALL;ALL;${GEN_TEMP}" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}"
echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} MATH_ALL [SWI]"
bash run_eval_lm.sh "1;${MODEL};1;MATH_ALL;ALL;${GEN_TEMP}" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}-swi"
echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} MATH_ALL [CoT]"
bash run_eval_lm.sh "1;${MODEL};1;MATH_ALL;ALL;${GEN_TEMP}" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}--cot"
echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} MATH_ALL [PS]"
bash run_eval_lm.sh "1;${MODEL};1;MATH_ALL;ALL;${GEN_TEMP}" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}--ps"
echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} MATH_ALL [ARR]"
bash run_eval_lm.sh "1;${MODEL};1;MATH_ALL;ALL;${GEN_TEMP}" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}--arr"
```

</details>


## Generalizability of Speaking with Intent (Paper Section XX)

### Multiple-Choice QA (Paper Table XX)

<details><summary>Experimental Settings</summary>

- **Datasets**: LogiQA, CSQA, SIQA, OBQA, ARC, BBH, MMLU, and MMLU-Pro
  - [x] `logiqa`: **LogiQA** - [Paper](https://arxiv.org/abs/2007.08124); [GitHub](https://github.com/lgw863/LogiQA-dataset)
  - [x] `commonsense_qa`: **CommonsenseQA** (CSQA) - [Paper](https://aclanthology.org/N19-1421/); [Dataset](https://huggingface.co/datasets/tau/commonsense_qa)
  - [x] `social_iqa`: **SocialIQA** (SIQA) - [Paper](https://arxiv.org/abs/1904.09728); [Dataset](https://huggingface.co/datasets/allenai/social_i_qa)
  - [x] `openbookqa`: **OpenBookQA** (OBQA) - [Paper](https://arxiv.org/abs/1809.02789); [Homepage](https://leaderboard.allenai.org/open_book_qa/submissions/get-started); [GitHub](https://github.com/allenai/OpenBookQA)
  - [x] `ai2_arc`: **ARC** - [Paper](https://arxiv.org/abs/1803.05457); [Homepage](https://leaderboard.allenai.org/arc/submissions/get-started); [GitHub](https://github.com/allenai/aristo-leaderboard)
  - [x] `bbh`: **BigBench Hard** (BBH) - [BigBench Paper](https://arxiv.org/abs/2206.04615); [BigBench GitHub](https://github.com/google/BIG-bench); [BBH Paper](https://arxiv.org/abs/2210.09261); [BBH Dataset](https://huggingface.co/datasets/lukaemon/bbh)
  - [x] `mmlu`: **MMLU** - [Paper](https://arxiv.org/abs/2009.03300); [Dataset](https://huggingface.co/datasets/cais/mmlu); [No-Train Data](https://huggingface.co/datasets/hails/mmlu_no_train)
  - [x] `mmlu_pro`: **MMLU-Pro** - [Paper](https://arxiv.org/abs/2406.01574); [Dataset](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- **Comparison**: (Zero-shot Settings)
  - [x] Baseline (LLM Generation w/o SWI)
  - [x] **SWI** (Ours): Require LLMs to speak with (their own) intent.
- **Models**:
  - [x] `meta-llama/Llama-3.1-8B-Instruct` ([Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))

</details>

<details><summary>Experiment Script</summary>

```bash
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
PROJECT_DIR="/path/to/SWI/"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
GEN_TEMP="0.0"
OUTPUT_DIR="${PROJECT_DIR}/results/swi_results-temp_${GEN_TEMP}"  # Baseline output directory

# [Reasoning & Answer Generation] **First**, freely generate answers with reasoning:
echo -e "\n\n >>> bash run_gen_lm.sh --hf_id ${MODEL} QA_ALL [Baseline]"
bash run_gen_lm.sh "1;${MODEL};1;QA_ALL;${GEN_TEMP};4096" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}"
echo -e "\n\n >>> bash run_gen_lm-swi.sh --hf_id ${MODEL} QA_ALL [SWI]"
bash run_gen_lm-swi.sh "1;${MODEL};1;QA_ALL;${GEN_TEMP};4096" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}-swi"

# [Answer Extraction & Evaluation] **Second**, extract the answers and evaluate them:
echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} QA_ALL [Baseline]"
bash run_eval_lm.sh "1;${MODEL};1;QA_ALL;ALL;${GEN_TEMP}" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}"
echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} QA_ALL [SWI]"
bash run_eval_lm.sh "1;${MODEL};1;QA_ALL;ALL;${GEN_TEMP}" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}-swi"
```

</details>

### Text Summarization (Paper Table XX)

<details><summary>Experimental Settings</summary>

- **Datasets**: CNN/DailyMail (CDM), XSum, XL-Sum, SAMSum, DialogSum, and WikiLingua
  - [x] `cnn_dailymail`: **CNN/DailyMail** (CDM) - [Paper1](https://aclanthology.org/P17-1099/), [Paper2](https://proceedings.neurips.cc/paper_files/paper/2015/hash/afdec7005cc9f14302cd0474fd0f3c96-Abstract.html); [Dataset](https://huggingface.co/datasets/abisee/cnn_dailymail)
  - [x] `xsum`: **XSum** - [Paper](https://aclanthology.org/D18-1206/); [Dataset](https://huggingface.co/datasets/EdinburghNLP/xsum)
  - [x] `xlsum`: **XL-Sum** - [Paper](https://aclanthology.org/2021.findings-acl.413/); [Dataset](https://huggingface.co/datasets/GEM/xlsum)
  - [x] `samsum`: **SAMSum** - [Paper](https://aclanthology.org/D19-5409/); [Dataset](https://huggingface.co/datasets/Samsung/samsum)
  - [x] `dialogsum`: **DialogSum** - [Paper](https://aclanthology.org/2021.findings-acl.449/); [Dataset](https://huggingface.co/datasets/knkarthick/dialogsum)
  - [x] `wiki_lingua`: **WikiLingua** - [Paper](https://aclanthology.org/2020.findings-emnlp.360/); [Dataset](https://huggingface.co/datasets/GEM/wiki_lingua)
- **Comparison**: (Zero-shot Settings)
  - [x] Baseline (LLM Generation w/o SWI)
  - [x] **SWI** (Ours): Require LLMs to speak with (their own) intent.
- **Models**:
  - [x] `meta-llama/Llama-3.1-8B-Instruct` ([Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))

</details>

<details><summary>Experiment Script</summary>

```bash
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
PROJECT_DIR="/path/to/SWI/"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
GEN_TEMP="0.0"
OUTPUT_DIR="${PROJECT_DIR}/results/swi_results-temp_${GEN_TEMP}"  # Baseline output directory

# [Reasoning & Answer Generation] **First**, freely generate answers with reasoning:
echo -e "\n\n >>> bash run_gen_lm.sh --hf_id ${MODEL} SUM_ALL [Baseline]"
bash run_gen_lm.sh "1;${MODEL};1;SUM_ALL;${GEN_TEMP};4096" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}"
echo -e "\n\n >>> bash run_gen_lm-swi.sh --hf_id ${MODEL} SUM_ALL [SWI]"
bash run_gen_lm-swi.sh "1;${MODEL};1;SUM_ALL;${GEN_TEMP};4096" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}-swi"

# [Answer Extraction & Evaluation] **Second**, extract the answers and evaluate them:
echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} SUM_ALL [Baseline]"
bash run_eval_lm.sh "1;${MODEL};1;SUM_ALL;ALL;${GEN_TEMP}" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}"
echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} SUM_ALL [SWI]"
bash run_eval_lm.sh "1;${MODEL};1;SUM_ALL;ALL;${GEN_TEMP}" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}-swi"
```

</details>

### Fact Checking of Summaries (Paper Table XX)

<details><summary>Experimental Settings</summary>

- **Datasets**: CNN/DailyMail (CDM), XSum, XL-Sum, SAMSum, DialogSum, and WikiLingua
- **Comparison**:
  - [x] Baseline (w/o SWI)
  - [x] **SWI** (Ours): Require LLMs to speak with (their own) intent.
- **Setting**:
  - Reference: [BottleHumor](https://arxiv.org/pdf/2502.18331) (Section 4.4)
  - Sample 100 data points from each summarization dataset.
  - Let GPT decompose the atomic facts in the candidate summary and the reference, 
  - and then compare the recall and precision of the fact coverage (against LLM hallucinations).
- **Models**:
  - [x] `gpt-4o-mini` API ([Link](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/))

</details>

<details><summary>Experiment Script</summary>

```bash
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
PROJECT_DIR="/path/to/SWI/"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
GEN_TEMP="0.0"
EVAL_NUM="100"
OUTPUT_DIR="${PROJECT_DIR}/results/swi_results-temp_${GEN_TEMP}"  # Baseline
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"  # Input your valid key here. We use "gpt-4o-mini" by default

echo -e "\n\n >>> bash run_eval_prf.sh --hf_id ${MODEL} SUM_ALL PRF [Baseline]"
bash run_eval_prf.sh "1;${MODEL};1;SUM_ALL;${GEN_TEMP};${EVAL_NUM}" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}" "${OPENAI_API_KEY}"
echo -e "\n\n >>> bash run_eval_prf.sh --hf_id ${MODEL} SUM_ALL PRF [SWI]"
bash run_eval_prf.sh "1;${MODEL};1;SUM_ALL;${GEN_TEMP};${EVAL_NUM}" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}-swi" "${OPENAI_API_KEY}"
```

</details>

## Human Evaluation of The Generated Intent (Paper Section XX)

<details><summary>Preparing Data for Human Evaluation</summary>

```bash
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
PROJECT_DIR="/path/to/SWI/"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
GEN_TEMP="0.0"
OUTPUT_DIR_BASELINE="${PROJECT_DIR}/results/swi_results-temp_${GEN_TEMP}"  # Baseline results
OUTPUT_DIR_SWI="${PROJECT_DIR}/results/swi_results-temp_${GEN_TEMP}-swi"  # SWI results (speaking with intent)

# We sample 12 data points per dataset and convert the JSON results into CSV for Human Evaluation.
# Each data point has 3 duplications and each of them is evaluate by different native English speaker. 420 in total
for TASK_TYPE in "QA_TWO" "MATH_TWO" "SUM_TWO"
do
  echo -e "\n\n >>> python3 run_human_eval_intent.py ${TASK_TYPE}"
  python3 run_human_eval_intent.py --verbose --task 1 --hf_id "${MODEL}" \
    --cache_dir "${CACHE_DIR}" --project_dir "${PROJECT_DIR}" \
    --output_dir "${OUTPUT_DIR_SWI}" \
    --min_doc_length 500 --max_doc_length 1000 \
    --num_item_per_task 12 --num_duplication 3 --num_item_in_a_row 6 --eval_task_name "${TASK_TYPE}"
done
```

</details>

The CSV files and HTML pages for human evaluation are under the [human_eval](./human_eval) directory.


## License

Please refer to the [LICENSE](./LICENSE) file for more details.

## Citation

* **Paper** (arXiv): https://arxiv.org/abs/2503.21544

```bibtex
@article{yin2025swi,
  title   = {SWI: Speaking with Intent in Large Language Models},
  author  = {Yin, Yuwei and Hwang, EunJeong and Carenini, Giuseppe},
  journal = {arXiv preprint arXiv:2503.21544},
  year    = {2025},
  url     = {https://arxiv.org/abs/2503.21544},
}
```

---
