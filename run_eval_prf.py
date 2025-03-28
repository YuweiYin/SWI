#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin" and "@eujhwang"
"""

import os
import time
import json
import string
import random
import logging
from typing import Optional, List

import fire
import numpy as np

import openai
from openai import AzureOpenAI

from utils.eval_utils import FACT_DECOM_PROMPT, FACT_INFER_PROMPT


class SummaryEval:

    def __init__(
            self,
            logger,
            verbose: bool = False,
            seed: int = 42,
            eval_task_name="samsum,dialogsum,wiki_lingua,cnn_dailymail,xsum,xlsum",
            cache_dir: Optional[str] = os.path.join(os.path.expanduser("~"), ".cache/huggingface"),
            results_dir: Optional[str] = "results/gen_eval/baseline/",  # or "results/gen_eval/swi/"
            openai_model: str = "gpt-4o-mini",
            openai_api_key: str = "YOUR_OPENAI_API_KEY",
            hf_id: Optional[str] = "meta-llama/Llama-3.1-8B-Instruct",
            eval_num: int = 10,
            # **kwargs
    ):
        self.verbose = verbose
        self.logger = logger
        self.seed = seed
        self.hf_id = hf_id
        self.hf_name = "--".join(hf_id.split("/"))
        self.cache_dir = cache_dir
        self.eval_num = eval_num

        self.eval_task_name = eval_task_name
        self.results_dir = results_dir

        # OpenAI settings
        # self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model  # "gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-mini", etc.
        try:
            assert isinstance(self.openai_api_key, str)
            self.client = AzureOpenAI(
                api_key=os.environ["OPENAI_KEY2"],
                api_version="2024-08-01-preview",
                azure_endpoint=os.environ["AZURE_END_POINT"]
            )
        except Exception as e:
            self.logger.info(f">>> !!! >>> Can NOT build the OpenAI agent: openai_api_key = {self.openai_api_key}\n{e}")
            self.client = None

        self.sum_task_set = {
            # Text Summarization
            "samsum", "dialogsum", "wiki_lingua",  # Num Items: SAMSum = 819; DialogSum = 1,500; WikiLingua = 3,000
            "cnn_dailymail", "xsum", "xlsum",  # Num Items: CDM = 11,490; XSum = 11,334; XL-Sum = 11,535
        }

        self.punc_remover = str.maketrans("", "", string.punctuation)  # r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        self.space_remover = str.maketrans("", "", string.whitespace)  # " \t\n\r\v\f"

    @staticmethod
    def _handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    def send_request(
            self,
            prompt: str,
            temperature: float = 0.1,
            max_tokens: int = 512,
    ):
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            time.sleep(2)
            return response.choices[0].message.content
        except openai.BadRequestError as e:
            if self.verbose:
                self.logger.info(f"Bad Request Error: {e}")
            return ""

    def decompose_sentences(
            self,
            paragraph: str,
    ):
        """
        Generates a decomposition of the given paragraph into atomic sentences.
        Returns a list of sentences, filtered against stop expressions.
        """
        prompt = FACT_DECOM_PROMPT.format(paragraph=paragraph)
        response = self.send_request(prompt, temperature=0.1, max_tokens=512)
        if not response:
            return []

        try:
            cleaned = response.replace("```python", "").replace("```", "").strip()
            decomposed = eval(cleaned)
        except Exception as e:
            decomposed = [line.strip() for line in response.split("\n") if line.strip().startswith('"')]
            if self.verbose:
                self.logger.info(e)

        return [fact for fact in decomposed if fact.strip()]

    def evaluate_prf(
            self,
            prompt: str,
            decomposed_sentences: list,
            full_sentence: str,
    ):
        """
        Each sentence in decomposed sentences will be compared with the information in full_sentence.
        Final output will be served as precision and recall
        """
        # if not decomposed_sentences:
        if not isinstance(decomposed_sentences, list) or len(decomposed_sentences) == 0:
            return -1, []

        score = 0
        answers = []
        for decomposed_sentence in decomposed_sentences:
            response = self.send_request(
                prompt=prompt.format(sentence1=decomposed_sentence, sentence2=full_sentence),
                temperature=0.1, max_tokens=3)
            if not response:
                return -1, []

            answers.append(response)
            if "yes" in response.lower():
                score += 1

        return score / len(decomposed_sentences), answers

    def run_evaluation(
            self,
            eval_task_name: str,
    ):
        assert eval_task_name in self.sum_task_set, f"AssertionError: task {eval_task_name} not in sum_task_set"

        # Load the generation outputs
        assert isinstance(self.results_dir, str) and os.path.isdir(self.results_dir), "Please specify --results_dir"
        results_dir = os.path.join(self.results_dir, eval_task_name, self.hf_name)
        results_fp = os.path.join(results_dir, "results_gen.json")
        assert os.path.isfile(results_fp), f"Assertion Error: results_fp does not exist: {results_fp}"
        with open(results_fp, "r", encoding="utf-8") as fp_in:
            results = json.load(fp_in)
        assert isinstance(results, dict)

        # Set the saving filepath
        eval_output_fp = os.path.join(results_dir, "results_eval_llm.json")
        if self.verbose:
            self.logger.info(f"Results will be saved at: {eval_output_fp}")

        # Deal with each subtask (if the dataset has multiple subtasks)
        all_scores = dict()  # The scores of the whole dataset (all subtasks)
        data_item_cnt_total = 0  # The total number of items we will deal with
        miss_final_cnt_total = 0  # The total number of items that miss "Final Summary:" in their outputs
        end_with_eot_cnt_total = 0  # The total number of items that end with end-of-text token
        show_cnt = 100
        for subtask_name, data_list in results.items():
            assert isinstance(subtask_name, str) and isinstance(data_list, list)
            len_subtask = len(data_list)
            assert len_subtask > 0
            if self.verbose:
                self.logger.info(f">>> [Subtask: {subtask_name}] # Items = {len_subtask}")

            # Deal with each data item in this subtask
            miss_final_cnt_st = 0  # The number of items in this subtask that miss "Final Summary:" in their outputs
            end_with_eot_cnt_st = 0  # The total number of items in this subtask that end with end-of-text token
            cur_st_results: List[dict] = []  # The evaluation results (dict) of the whole subtask
            invalid_summary_instances = 0
            for idx, cur_data_dict in enumerate(data_list):
                assert isinstance(cur_data_dict, dict)
                cur_res_dict = dict()  # The evaluation results of the current data item

                self.logger.info("len(cur_st_results):", len(cur_st_results), "[eval_num]:", self.eval_num)
                if len(cur_st_results) >= self.eval_num:
                    break

                # Load the model prediction (summary) and golden references of the data item
                prediction = str(cur_data_dict["output_text"]).strip()  # model's prediction to evaluate
                references = cur_data_dict["answers"]  # The list golden references (usually the length is 1)
                references = [str(_ref).strip() for _ref in references]

                # Load other attributes of the data item
                info = cur_data_dict["info"]  # task-specific information
                assert isinstance(info, dict) and "task_type" in info
                task_type = info["task_type"]  # The task type ("sum" means summarization)
                assert task_type == "sum" and "article" in info

                if "end_with_eot" in cur_data_dict:
                    end_with_eot = bool(cur_data_dict["end_with_eot"])  # True if the output ends with end-of-text
                    cur_res_dict["end_with_eot"] = end_with_eot
                    if end_with_eot:
                        end_with_eot_cnt_st += 1
                        end_with_eot_cnt_total += 1

                # Extract the final answer from the generated output
                if "Final Summary:" in prediction:
                    pred_final = prediction.split("Final Summary:")[-1].strip()
                else:
                    self.logger.info("Skip! -- Final Summary does not exist..")
                    invalid_summary_instances += 1
                    continue

                if not pred_final or not pred_final.endswith("."):
                    # skip invalid final summary. if final summary did not end with ".", then its likely invalid
                    self.logger.info("Skip! -- Final Summary does not exist..")
                    invalid_summary_instances += 1
                    continue

                cur_res_dict["eval_score"] = dict()

                # calculate precision score
                decomposed_prediction = self.decompose_sentences(pred_final)
                precision, precision_answers = self.evaluate_prf(
                    FACT_INFER_PROMPT, decomposed_prediction, "\n".join(references))
                if precision == -1:
                    self.logger.info("Skip! -- Invalid precision...")
                    continue

                self.logger.info(f"[precision]: {precision}")
                self.logger.info("[reference]:\n" + "\n".join(references))
                for x, y in zip(decomposed_prediction, precision_answers):
                    self.logger.info(f"[{y}]: [{x}]")

                # calculate recall score
                decomposed_reference = self.decompose_sentences("\n".join(references))
                recall, recall_answers = self.evaluate_prf(FACT_INFER_PROMPT, decomposed_reference, pred_final)

                self.logger.info(f"[recall]: {recall}")
                if recall == -1:
                    self.logger.info("Skip! -- Invalid recall...")
                    continue

                self.logger.info(f"[prediction]:\n {pred_final}")
                for x, y in zip(decomposed_reference, precision_answers):
                    self.logger.info(f"[{y}]: [{x}]")

                # calculate f1 score
                f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
                self.logger.info(f"[f1_score]: {f1_score}")

                # results to be saved
                cur_res_dict["prediction"] = pred_final
                cur_res_dict["reference"] = references
                cur_res_dict["decomposed_prediction"] = decomposed_prediction
                cur_res_dict["decomposed_prediction_recall_answers"] = recall_answers
                cur_res_dict["decomposed_reference"] = decomposed_reference
                cur_res_dict["decomposed_reference_precision_answers"] = precision_answers
                cur_res_dict["eval_score"] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1_score,
                }

                # After the evaluation of the current data item, save the current results and show logs
                cur_st_results.append(cur_res_dict)
                if self.verbose and len(cur_st_results) % show_cnt == 0:
                    self.logger.info(f">>> Progress: [{subtask_name}] [{len(cur_st_results)} / {len_subtask}] "
                                     f"[end_with_eot_cnt_st = {end_with_eot_cnt_st}]"
                                     f"[miss_final_cnt_st = {miss_final_cnt_st}]")

            # Save the results (List[dict]), scores (List[float]), and score statistics (`num_items` and `score_avg`)
            avg_precision, avg_recall, avg_f1 = [], [], []
            for item in cur_st_results:
                eval_score = item["eval_score"]
                p = eval_score["precision"]
                r = eval_score["recall"]
                f1 = eval_score["f1"]
                avg_precision.append(p)
                avg_recall.append(r)
                avg_f1.append(f1)

            num_items_st = len(cur_st_results)
            all_scores[subtask_name] = {
                "results": cur_st_results,
                "num_items": num_items_st,
                "avg_precision": float(np.mean(avg_precision).item()),
                "avg_recall": float(np.mean(avg_recall).item()),
                "avg_f1": float(np.mean(avg_f1).item()),
            }
            data_item_cnt_total += num_items_st
            if self.verbose:
                self.logger.info(f">>> Done Subtask. [{subtask_name}] "
                                 f"[data_item_cnt_st = {num_items_st}; "
                                 f"avg_precision = {avg_precision}]; "
                                 f"avg_recall = {avg_recall}]; avg_f1 = {avg_f1}] "
                                 f"[end_with_eot_cnt_st = {end_with_eot_cnt_st}] "
                                 f"[miss_final_cnt_st = {miss_final_cnt_st}]\n")

        # Compute the overall score statistics of different metrics and show stats
        total_num_items = 0
        total_precisions = float(0.0)
        total_recalls = float(0.0)
        total_f1s = float(0.0)
        for subtask_name, subtask_results in all_scores.items():
            assert isinstance(subtask_results, dict)
            num_items = int(subtask_results["num_items"])
            avg_precision = subtask_results["avg_precision"]
            avg_recall = subtask_results["avg_recall"]
            avg_f1 = subtask_results["avg_f1"]
            total_num_items += num_items
            total_precisions += avg_precision * num_items
            total_recalls += avg_recall * num_items
            total_f1s += avg_f1 * num_items

        overall_precision_avg = total_precisions / total_num_items if total_num_items > 0 else 0.0
        overall_recall_avg = total_recalls / total_num_items if total_num_items > 0 else 0.0
        overall_f1_avg = total_f1s / total_num_items if total_num_items > 0 else 0.0
        self.logger.info(f">>> Overall Scores: (num_items = {total_num_items}) "
                         f"overall_precision_avg = {overall_precision_avg:.5f} | "
                         f"overall_recall_avg = {overall_recall_avg:.5f} | "
                         f"overall_f1_avg = {overall_f1_avg:.5f}")
        overall_score = {
            "total_num_items": total_num_items,
            "overall_precision_avg": overall_precision_avg,
            "overall_recall_avg": overall_recall_avg,
            "overall_f1_avg": overall_f1_avg,
        }
        all_scores["overall_score"] = overall_score
        self.logger.info(f"[all_scores - overall_score]:\n{overall_score}")
        self.logger.info(f">>> DONE ALL. [Task: {eval_task_name}] "
                         f"[# Data items in total = {data_item_cnt_total}] "
                         f"[# Items end with eot = {end_with_eot_cnt_total}] "
                         f"[# Effective data items = {data_item_cnt_total - miss_final_cnt_total}] "
                         f"[# Missing Final Answer = {miss_final_cnt_total}]\n")

        # Save the generation outputs
        dumped = json.dumps(
            all_scores,
            indent=2,  # indent=None,
            default=self._handle_non_serializable,
            ensure_ascii=True,
        )
        with open(eval_output_fp, "w", encoding="utf-8") as fp_out:
            fp_out.write(dumped)
        self.logger.info(
            f">>> hf_id = {self.hf_id}; eval_output_fp: {os.path.abspath(eval_output_fp)}\n"
        )


def main(
    eval_task_name="ALL",  # "samsum,dialogsum,wiki_lingua,cnn_dailymail,xsum,xlsum"
    results_dir: Optional[str] = "results/gen_eval/baseline/",  # or "results/gen_eval/swi/"
    hf_id: Optional[str] = "meta-llama/Llama-3.1-8B-Instruct",
    cache_dir: Optional[str] = os.path.join(os.path.expanduser("~"), ".cache/huggingface"),
    openai_model: str = "gpt-4o-mini",
    openai_api_key: str = "YOUR_OPENAI_API_KEY",
    seed: int = 42,
    verbose: bool = False,
    eval_num: int = 10,
    **kwargs
) -> None:
    """
    Compute the LLM-based precision and recall scores for the summaries
      generated by Baseline method and our SWI method, based on the golden reference summaries.
    Reference: https://arxiv.org/abs/2502.18331  [Section 4.4]

    :param eval_task_name: The name(s) of the evaluation task. (e.g., "samsum", "dialogsum", and "samsum,dialogsum")
    :param results_dir: The file path to the generated summaries and golden references.
    :param hf_id: ORGANIZATION_NAME/MODEL_NAME, e.g., "meta-llama/Llama-3.1-8B-Instruct"
    :param cache_dir: The root directory of the Hugging Face cache.
    :param openai_model: e.g., "gpt-4o", "gpt-4o-2024-08-06", and "gpt-4o-mini"
    :param openai_api_key: your valid OpenAI API Key. https://platform.openai.com/
    :param seed: Random seed of all modules.
    :param verbose: Verbose mode: show logs.
    :param eval_num: The number of instances to be evaluated.
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger("Fact_Checking")

    random.seed(seed)
    np.random.seed(seed)

    if isinstance(kwargs, dict):
        logger.info(f">>> Unused parameters in kwargs: {kwargs}")

    if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
        os.environ["HF_HOME"] = cache_dir
        # os.environ["HF_HOME"] = os.path.join(cache_dir, "datasets")
        # os.environ["HF_HOME"] = os.path.join(cache_dir, "hub")
    else:
        cache_dir = None

    # We evaluate the summary quality of the Baseline method and our SWI method
    eval_task_name = str(eval_task_name).strip()
    if eval_task_name == "ALL" or eval_task_name == "SUM_ALL":
        eval_task_name = "samsum,dialogsum,wiki_lingua,cnn_dailymail,xsum,xlsum".split(",")
    if "," in eval_task_name:
        eval_task_name = eval_task_name.split(",")

    lm_eval = SummaryEval(
        eval_task_name=eval_task_name,
        results_dir=results_dir,
        cache_dir=cache_dir,
        openai_model=openai_model,
        openai_api_key=openai_api_key,
        seed=seed,
        logger=logger,
        hf_id=hf_id,
        verbose=verbose,
        eval_num=int(eval_num),
    )

    if isinstance(eval_task_name, tuple) or isinstance(eval_task_name, list):  # Deal with a list of eval tasks
        for cur_task_name in eval_task_name:
            assert cur_task_name in lm_eval.sum_task_set, f"AssertionError: {cur_task_name} not in sum_task_set"
            cur_task_name = str(cur_task_name).strip()
            logger.info(f">>> <START> {cur_task_name}\n")
            lm_eval.run_evaluation(eval_task_name=cur_task_name)
            logger.info(f">>> <END> {cur_task_name}\n\n\n")
    elif isinstance(eval_task_name, str):  # Deal with a single eval task
        assert eval_task_name in lm_eval.sum_task_set, f"AssertionError: task name {eval_task_name} not in sum_task_set"
        eval_task_name = str(eval_task_name).strip()
        logger.info(f">>> <START> {eval_task_name}\n")
        lm_eval.run_evaluation(eval_task_name=eval_task_name)
        logger.info(f">>> <END> {eval_task_name}\n\n\n")
    else:
        raise ValueError(f"--eval_task_name should be a tuple/list/str: {eval_task_name}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
