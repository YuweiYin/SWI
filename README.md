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
Intent, typically clearly formulated and planned, functions as a cognitive framework 
for communication and problem-solving. This paper introduces the concept of Speaking 
with Intent (SWI) in large language models (LLMs), where the explicitly generated intent 
encapsulates the model's underlying intention and provides high-level planning to guide 
subsequent analysis and action. By emulating deliberate and purposeful thoughts in the 
human mind, SWI is hypothesized to enhance the reasoning capabilities and generation quality 
of LLMs. Extensive experiments on text summarization, multi-task question answering, and 
mathematical reasoning benchmarks consistently demonstrate the effectiveness and 
generalizability of Speaking with Intent over direct generation without explicit intent. 
Further analysis corroborates the generalizability of SWI under different experimental 
settings. Moreover, human evaluations verify the coherence, effectiveness, and interpretability 
of the intent produced by SWI. The promising results in enhancing LLMs with explicit intents 
pave a new avenue for boosting LLMs' generation and reasoning abilities with cognitive notions.
```

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

## Datasets and Models

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

## Experiments

For each bash script, please ensure `CACHE_DIR` and `PROJECT_DIR` in the script are 
correct Hugging Face cache directory (default: `"~/.cache/huggingface/"`) and 
project root directory (`"/path/to/SWI/"`).

```bash
mkdir -p logs/  # where we save running logs
mkdir -p results/  # where we save experimental results
```

### Main Results

<details><summary>Experiment Script</summary>

```bash
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
PROJECT_DIR="/path/to/SWI/"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="${PROJECT_DIR}/results/results"

# [Reasoning & Answer Generation] **First**, freely generate answers with reasoning:
echo -e "\n\n >>> bash run_gen_lm.sh --hf_id ${MODEL} ALL [DA]"  # Direct Answer (DA) - baseline
bash run_gen_lm.sh "1;42;${MODEL};1;ALL;0.0;4096" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}--da"
echo -e "\n\n >>> bash run_gen_lm-swi.sh --hf_id ${MODEL} ALL [SWI]"  # Speaking with Intent (SWI) - ours
bash run_gen_lm-swi.sh "1;42;${MODEL};1;ALL;0.0;4096" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}--swi"

# [Answer Extraction & Evaluation] **Second**, extract the answers and evaluate them:
echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} ALL [Baseline]"
bash run_eval_lm.sh "1;42;${MODEL};1;ALL;ALL;0.0" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}--da"
echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} ALL [SWI]"
bash run_eval_lm.sh "1;42;${MODEL};1;ALL;ALL;0.0" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}--swi"
```

</details>

### Fact Checking of Summaries

<details><summary>Experimental Settings</summary>

- **Datasets**: CNN/DailyMail (CDM), XSum, XL-Sum, DialogSum, and WikiLingua
- **Comparison**:
  - [x] DA: Direct Answer (w/o SWI)
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
OUTPUT_DIR="${PROJECT_DIR}/results/results"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"  # Input your valid key here. We use "gpt-4o-mini" by default
EVAL_NUM="100"

echo -e "\n\n >>> bash run_eval_prf.sh --hf_id ${MODEL} SUM_ALL PRF [Baseline]"
bash run_eval_prf.sh "1;${MODEL};1;SUM_ALL;0.0;${EVAL_NUM}" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}--da" "${OPENAI_API_KEY}"
echo -e "\n\n >>> bash run_eval_prf.sh --hf_id ${MODEL} SUM_ALL PRF [SWI]"
bash run_eval_prf.sh "1;${MODEL};1;SUM_ALL;0.0;${EVAL_NUM}" "${CACHE_DIR}" "${PROJECT_DIR}" "${OUTPUT_DIR}--swi" "${OPENAI_API_KEY}"
```

</details>

## Human Evaluation of The Generated Intent

<details><summary>Preparing Data for Human Evaluation</summary>

```bash
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
PROJECT_DIR="/path/to/SWI/"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="${PROJECT_DIR}/results/results"

# We sample 12 data points per dataset and convert the JSON results into CSV for Human Evaluation.
# Each data point has 3 duplications and each of them is evaluate by different native English speaker. 420 in total
for TASK_TYPE in "QA_TWO" "MATH_TWO" "SUM_TWO"
do
  echo -e "\n\n >>> python3 run_human_eval_intent.py ${TASK_TYPE}"
  python3 run_human_eval_intent.py --verbose --task 1 --hf_id "${MODEL}" \
    --cache_dir "${CACHE_DIR}" --project_dir "${PROJECT_DIR}" \
    --output_dir "${OUTPUT_DIR}--swi" \
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
