#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import re
import time
import json
import string
from typing import Optional, List

import fire
import numpy as np
import pandas as pd

from datasets import Dataset

# Mathematical Reasoning (Math)
from tasks.gsm8k import EvalTaskGSM8K
from tasks.gsm8k_platinum import EvalTaskGSM8KPlatinum
from tasks.math500 import EvalTaskMATH500
from tasks.amc23 import EvalTaskAMC23
from tasks.aime24 import EvalTaskAIME24
from tasks.aime25 import EvalTaskAIME25

# Multiple-choice Question Answering (MCQA)
from tasks.logiqa import EvalTaskLogiQA
from tasks.commonsense_qa import EvalTaskCommonsenseQA
from tasks.social_iqa import EvalTaskSocialIQA
from tasks.openbookqa import EvalTaskOpenbookQA
from tasks.ai2_arc import EvalTaskAi2Arc
from tasks.bbh import EvalTaskBbh
from tasks.mmlu import EvalTaskMmlu
from tasks.mmlu_pro import EvalTaskMmluPro

# Text Summarization (Sum)
from tasks.cnn_dailymail import EvalTaskCnnDailymail
from tasks.xsum import EvalTaskXSum
from tasks.xlsum import EvalTaskXlSum
from tasks.samsum import EvalTaskSamSum
from tasks.dialogsum import EvalTaskDialogSum
from tasks.wiki_lingua import EvalTaskWikiLingua

from utils.init_functions import logger_setup, cuda_setup, random_setup


class HumanEvalIntent:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: dict,
            seed: int = 42,
            eval_task_name="",
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
            hf_id: str = "meta-llama/Llama-3.1-8B-Instruct",
            debug: bool = False,
            output_dir: Optional[str] = None,
            num_item_per_task: int = 10,
            min_doc_length: int = 500,
            max_doc_length: int = 1000,
    ):
        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.seed = seed
        self.hf_id = hf_id
        self.hf_name = "--".join(hf_id.split("/"))
        self.debug = debug

        if isinstance(project_dir, str) and os.path.isdir(project_dir):
            self.project_dir = project_dir
        else:
            self.project_dir = os.getcwd()
        assert os.path.isdir(project_dir)

        self.eval_task_name = eval_task_name
        self.output_dir = output_dir
        self.num_item_per_task = num_item_per_task
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length

        # Cache directory
        self.home_dir = os.path.expanduser("~")
        if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(self.home_dir, ".cache/huggingface")
            # self.cache_dir = os.path.join(self.project_dir, ".cache/huggingface/")
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
        if self.verbose:
            self.logger.info(f">>> cache_dir: {self.cache_dir}")

        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        os.environ["HF_HOME"] = self.cache_dir
        self.model_path = os.path.join(
            self.cache_dir, "models--" + self.hf_name, "snapshots/model")
        assert os.path.isdir(self.model_path), f"AssertionError: assert os.path.isdir({self.model_path})"

        self.task_class_dict = {
            # Mathematical Reasoning (Math)
            "gsm8k": EvalTaskGSM8K,
            "gsm8k_platinum": EvalTaskGSM8KPlatinum,
            "math500": EvalTaskMATH500,
            "amc23": EvalTaskAMC23,
            "aime24": EvalTaskAIME24,
            "aime25": EvalTaskAIME25,
            # Multiple-choice Science Question Answering (MCQA)
            "logiqa": EvalTaskLogiQA,
            "commonsense_qa": EvalTaskCommonsenseQA,
            "social_iqa": EvalTaskSocialIQA,
            "openbookqa": EvalTaskOpenbookQA,
            "ai2_arc": EvalTaskAi2Arc,
            "bbh": EvalTaskBbh,
            "mmlu": EvalTaskMmlu,
            "mmlu_pro": EvalTaskMmluPro,
            # Text Summarization (Sum)
            "cnn_dailymail": EvalTaskCnnDailymail,
            "xsum": EvalTaskXSum,
            "xlsum": EvalTaskXlSum,
            "samsum": EvalTaskSamSum,
            "dialogsum": EvalTaskDialogSum,
            "wiki_lingua": EvalTaskWikiLingua,
        }
        self.math_set = {"gsm8k", "gsm8k_platinum", "math500", "amc23", "aime24", "aime25"}
        self.mcqa_set = {"logiqa", "commonsense_qa", "social_iqa", "openbookqa", "ai2_arc", "bbh", "mmlu", "mmlu_pro"}
        self.sum_set = {"cnn_dailymail", "xsum", "xlsum", "samsum", "dialogsum", "wiki_lingua"}

        self.punc_list = [ch for ch in string.punctuation]
        self.space_list = [ch for ch in string.whitespace]
        self.punc_set = set(self.punc_list)
        self.space_set = set(self.space_list)
        self.punc_remover = str.maketrans("", "", string.punctuation)  # r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        self.space_remover = str.maketrans("", "", string.whitespace)  # " \t\n\r\v\f"
        self.re_math = re.compile(r"(\$.*?\$)")  # Match math expressions

    @staticmethod
    def _handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    def string_clean(
            self,
            raw_str: str,
    ):
        raw_str = raw_str.strip()
        raw_str = raw_str.encode("utf-8", "ignore").decode("utf-8").strip()

        clean_str = ""
        dirty_flag = False
        for ch in raw_str:
            if ch.isalnum() or ch in self.punc_set or ch in self.space_set:
                clean_str += ch
            else:
                dirty_flag = True

        return clean_str, dirty_flag

    def extract_intent_list(
            self,
            prediction: str,
            delimiter: str = "Final Summary:",
    ):
        # Extract the intent and sentences (the full summary) from the prediction
        pred_split = prediction.split(delimiter)
        pred_final = pred_split[-1].strip()
        reasoning = "Final Summary:".join(pred_split[:-1])

        intent_list = []
        sentence_list = []
        reasoning_split = reasoning.split("</INTENT>")
        try:
            for piece in reasoning_split:
                piece = piece.strip()
                if "<INTENT>" in piece:
                    piece_split = piece.split("<INTENT>")
                    if len(piece_split) == 1:
                        intent_list.append(piece_split[0].strip())
                    else:
                        assert len(piece_split) == 2
                        _p0, _p1 = piece_split[0].strip(), piece_split[1].strip()
                        if len(_p1) > 0:
                            intent_list.append(_p1)
                        if len(_p0) > 0:
                            sentence_list.append(_p0)
        except AssertionError as e:
            if self.verbose:
                self.logger.info(f">>> !!! >>> {e}")
            return [], [], ""

        return intent_list, sentence_list, pred_final

    def compute_exact_match(
            self,
            pred_final: str,
            references: List[str],
            # **kwargs
    ) -> bool:
        pred_final = str(pred_final).strip()
        references = [str(_ref).strip() for _ref in references]

        # Matching anyone in the references will have an EM score of 1; otherwise 0.
        for ref in references:
            if pred_final == ref:
                return True

        # Normalize strings and then match
        pred_final_new, references_new = pred_final, references
        pred_final_new = pred_final_new.translate(self.punc_remover).strip()  # Remove all punctuations
        pred_final_new = pred_final_new.translate(self.space_remover).strip()  # Remove all whitespaces
        references_new = [_ref.translate(self.punc_remover).strip() for _ref in references_new]
        references_new = [_ref.translate(self.space_remover).strip() for _ref in references_new]
        for ref in references_new:
            if pred_final_new == ref:
                return True

        return False

    def do_convert(
            self,
            eval_task_name: str,
    ) -> dict:
        # Convert the result JSON files to CSV for Human evaluation.
        # One item in a CSV row

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load the generation outputs
        assert isinstance(self.output_dir, str) and os.path.isdir(self.output_dir), \
            "Please specify --output_dir"
        output_dir = os.path.join(self.output_dir, eval_task_name, self.hf_name)
        output_fp = os.path.join(output_dir, "results_gen.json")
        assert os.path.isfile(output_fp), f"Error: output_fp does not exist: {output_fp}"
        with open(output_fp, "r", encoding="utf-8") as fp_in:
            gen_results_json = json.load(fp_in)

        assert eval_task_name in self.task_class_dict, \
            f"AssertionError: task name {eval_task_name} not in task_class_dict"
        eval_task_class = self.task_class_dict[eval_task_name]

        eval_task_obj = eval_task_class(
            verbose=self.verbose,
            logger=self.logger,
            cache_dir=self.cache_dir,
            project_dir=self.project_dir,
        )

        self.logger.info(f">>> Evaluation Task: {eval_task_name}")
        task_info = eval_task_obj.load_task()
        dataset_list = task_info["data"]

        # re_intent = re.compile(r"<INTENT>(.*?)</INTENT>")  # Match the intents
        # re_sentence = re.compile(r"</INTENT>(.*?)<INTENT>")  # Match sentences between intents
        re_cjk = re.compile(r"[\u4e00-\u9fff]+")  # Match CJK unicode characters

        all_results = {
            "task_name": [],
            "pred_final": [],
            "reasoning": [],
            "reference": [],
        }
        # Deal with each task (and sub-tasks)
        save_item_cnt_total = 0
        data_item_cnt_total = 0
        end_with_eot_cnt_total = 0
        show_cnt = 100
        ord_A = ord("A")
        dedup_set = set()  # Avoid duplication (e.g., the article of multiple data points in DialogSum may be the same)
        finish_flag = False
        for dataset_dict in dataset_list:
            if finish_flag:
                break

            ds_name, subset = dataset_dict["hf_dataset"], dataset_dict["hf_subset"]
            if ds_name == "allenai/ai2_arc" and subset != "ARC-Challenge":
                continue  # Only deal with the "ARC-Challenge" subtask in the ARC dataset
            if ds_name == "lukaemon/bbh" and subset != "causal_judgement":
                continue  # Only deal with the "causal_judgement" subtask in the BBH dataset
            if ds_name == "hails/mmlu_no_train" and subset != "elementary_mathematics":
                continue  # Only deal with the "elementary_mathematics" subtask in the MMLU dataset
            eval_split, eval_dataset = dataset_dict["eval_split"], dataset_dict["eval_dataset"]
            assert isinstance(eval_dataset, Dataset)
            len_dataset = len(eval_dataset)
            assert isinstance(ds_name, str) and len(ds_name) > 0
            if isinstance(subset, str) and len(subset) > 0:
                ds_id = f"{ds_name}---{subset}"
            else:
                ds_id = ds_name
            if self.verbose:
                self.logger.info(f">>> [Dataset: {ds_id}] [Eval: {eval_split}] # = {len_dataset}")

            assert ds_id in gen_results_json
            cur_results_list = gen_results_json[ds_id]
            assert isinstance(cur_results_list, list) and len(cur_results_list) == len_dataset > 0

            data_item_cnt_ds = 0
            for idx, cur_res_dict in enumerate(cur_results_list):
                data_item_cnt_ds += 1
                data_item_cnt_total += 1
                if finish_flag:
                    break

                # Load the attributes of the data item
                prediction = str(cur_res_dict["output_text"]).strip()  # model prediction to evaluate
                references = cur_res_dict["answers"]  # golden references (correct answers)
                references = [str(_ref).replace("\n", " ").strip() for _ref in references]
                reference = references[0]  # Only use the first reference
                info = cur_res_dict["info"]  # task-specific information
                end_with_eot = bool(cur_res_dict["end_with_eot"])  # True: output ends with end-of-text
                if end_with_eot:
                    end_with_eot_cnt_total += 1
                else:
                    if self.verbose and data_item_cnt_total % show_cnt == 0:
                        self.logger.info(f">>> Progress: [{ds_id}] [{data_item_cnt_total} / {len_dataset}]")
                    continue

                # Extract the final answer from the generated output
                assert isinstance(info, dict) and "task_type" in info
                task_type = info["task_type"]
                match task_type:
                    case "sum":
                        assert "article" in info
                        article = str(info["article"]).replace("\n", " ").strip()
                        article = article.replace("\r", "").strip()

                        article, article_dirty = self.string_clean(article)
                        prediction, prediction_dirty = self.string_clean(prediction)

                        if eval_task_name == "dialogsum":
                            article = article.replace("#Person1#:", "\n#Person1#:").strip()
                            article = article.replace("#Person2#:", "\n#Person2#:").strip()
                            article = article.replace("\n", "<br/>").strip()

                        if article_dirty or prediction_dirty:
                            continue
                        if len(article) > self.max_doc_length or len(article) < self.min_doc_length:
                            continue
                        if article not in dedup_set:
                            dedup_set.add(article)
                        else:
                            continue
                        if "Final Summary:" in prediction and "<INTENT>" in prediction and "</INTENT>" in prediction:
                            pred_split = prediction.split("Final Summary:")
                            pred_final = pred_split[-1].strip()
                            if "answer is" in pred_final:
                                pred_final = pred_final.split("answer is")[-1].strip()

                            # Instead of extracting each intent, we reform the raw text for better presentation
                            reasoning = "Final Summary:".join(pred_split[:-1])
                            reasoning = reasoning.replace("\n", " ").strip()
                            reasoning = reasoning.replace("<INTENT>", "\n<INTENT>").strip()
                            reasoning = reasoning.replace("</INTENT>", "</INTENT>\n").strip()
                            reasoning = reasoning.replace("\n\n", "\n").strip()

                            # HTML style
                            reasoning = reasoning.replace("\n", "<br/>").strip()
                            # reasoning = reasoning.replace("<INTENT>", "<INTENT style='color: red'>").strip()

                            if len(reasoning) > self.max_doc_length:
                                continue
                            intent_list = re.findall(r"<INTENT>.*?</INTENT>", reasoning)
                            if len(intent_list) < 2:
                                continue

                            # Filter out specific dirty samples
                            if article.startswith("(CNN)Blinky and Pinky on the Champs Elysees"):
                                continue

                            all_text = reference + reasoning + pred_final + article
                            cjk_list = re.findall(re_cjk, all_text)
                            if len(cjk_list) > 0:
                                continue  # Ignore instances containing CJK unicode characters
                            if "\"" in all_text:
                                continue

                            all_results["task_name"].append(eval_task_name)
                            all_results["pred_final"].append(pred_final)
                            all_results["reasoning"].append(reasoning)
                            all_results["reference"].append(reference)

                            if "article" in all_results:
                                all_results["article"].append(article)
                            else:
                                all_results["article"] = [article]

                            save_item_cnt_total += 1
                            if save_item_cnt_total >= self.num_item_per_task:
                                finish_flag = True
                                break
                        else:
                            continue
                    case "qa":
                        assert "context" in info and "question" in info and "options" in info and "ans_idx" in info
                        context = str(info["context"]).replace("\n", " ").strip()
                        context = context.replace("\r", "").strip()
                        question = str(info["question"]).replace("\n", " ").strip()
                        question = question.replace("\r", "").strip()
                        options, ans_idx = info["options"], info["ans_idx"]

                        context, context_dirty = self.string_clean(context)
                        question, question_dirty = self.string_clean(question)
                        prediction, prediction_dirty = self.string_clean(prediction)
                        if context_dirty or question_dirty or prediction_dirty:
                            continue
                        if len(context) + len(question) > self.max_doc_length:
                            continue
                        if question not in dedup_set:
                            dedup_set.add(question)
                        else:
                            continue
                        if not (isinstance(options, list) and len(options) > 0 and isinstance(ans_idx, int)
                                and 0 <= ans_idx < len(options)):
                            continue
                        if "$" in prediction or "$" in context or "$" in question:
                            continue  # Not going to deal with inline math expressions on HTML
                        if "Final Answer:" in prediction and "<INTENT>" in prediction and "</INTENT>" in prediction:
                            pred_split = prediction.split("Final Answer:")
                            pred_final = pred_split[-1].strip()
                            if "answer is" in pred_final:
                                pred_final = pred_final.split("answer is")[-1].strip()
                            if not self.compute_exact_match(pred_final, references):
                                continue

                            reasoning = "Final Answer:".join(pred_split[:-1])
                            reasoning = reasoning.replace("\n", " ").strip()
                            reasoning = reasoning.replace("<INTENT>", "\n<INTENT>").strip()
                            reasoning = reasoning.replace("</INTENT>", "</INTENT>\n").strip()
                            reasoning = reasoning.replace("\n\n", "\n").strip()

                            # HTML style
                            reasoning = reasoning.replace("\n", "<br/>").strip()
                            # reasoning = reasoning.replace("<INTENT>", "<INTENT style='color: red'>").strip()
                            # reasoning = self.convert_math_symbols(reasoning)

                            if len(reasoning) > self.max_doc_length:
                                continue
                            intent_list = re.findall(r"<INTENT>.*?</INTENT>", reasoning)
                            if len(intent_list) < 2:
                                continue

                            options_str = "<br/>".join(
                                [f"({chr(ord_A + _idx)}) {_op}" for _idx, _op in enumerate(options)])  # for HTML
                            if len(context) > 0:
                                question = f"Context: {context}\n\nQuestion: {question}"

                            all_text = reference + reasoning + pred_final + question + options_str
                            cjk_list = re.findall(re_cjk, all_text)
                            if len(cjk_list) > 0:
                                continue  # Ignore instances containing CJK unicode characters
                            if "\"" in all_text:
                                continue

                            all_results["task_name"].append(eval_task_name)
                            all_results["pred_final"].append(pred_final)
                            all_results["reasoning"].append(reasoning)
                            all_results["reference"].append(f"({chr(ord_A + ans_idx)}) {options[ans_idx]}")

                            if "question" in all_results:
                                all_results["question"].append(question)
                            else:
                                all_results["question"] = [question]

                            if "options" in all_results:
                                # all_results["options"].append(options)
                                all_results["options"].append(options_str)
                            else:
                                # all_results["options"] = [options]
                                all_results["options"] = [options_str]

                            save_item_cnt_total += 1
                            if save_item_cnt_total >= self.num_item_per_task:
                                finish_flag = True
                                break
                        else:
                            continue
                    case "math":
                        assert "question" in info and "solution" in info
                        question = str(info["question"]).replace("\n", " ").strip()
                        question = question.replace("\r", "").strip()

                        question, question_dirty = self.string_clean(question)
                        prediction, prediction_dirty = self.string_clean(prediction)
                        if question_dirty or prediction_dirty:
                            continue
                        if len(question) > self.max_doc_length:
                            continue
                        if question not in dedup_set:
                            dedup_set.add(question)
                        else:
                            continue
                        if "$" in prediction or "$" in question:
                            continue  # Not going to deal with inline math expressions on HTML
                        if "Final Answer:" in prediction and "<INTENT>" in prediction and "</INTENT>" in prediction:
                            pred_split = prediction.split("Final Answer:")
                            pred_final = pred_split[-1].strip()
                            if "answer is" in pred_final:
                                pred_final = pred_final.split("answer is")[-1].strip()
                            if not self.compute_exact_match(pred_final, references):
                                continue

                            reasoning = "Final Answer:".join(pred_split[:-1])
                            reasoning = reasoning.replace("\n", " ").strip()
                            reasoning = reasoning.replace("<INTENT>", "\n<INTENT>").strip()
                            reasoning = reasoning.replace("</INTENT>", "</INTENT>\n").strip()
                            reasoning = reasoning.replace("\n\n", "\n").strip()

                            # HTML style
                            reasoning = reasoning.replace("\n", "<br/>").strip()
                            # reasoning = reasoning.replace("<INTENT>", "<INTENT style='color: red'>").strip()
                            # reasoning = self.convert_math_symbols(reasoning)

                            if len(reasoning) > self.max_doc_length:
                                continue
                            intent_list = re.findall(r"<INTENT>.*?</INTENT>", reasoning)
                            if len(intent_list) < 2:
                                continue

                            all_text = reference + reasoning + pred_final + question
                            cjk_list = re.findall(re_cjk, all_text)
                            if len(cjk_list) > 0:
                                continue  # Ignore instances containing CJK unicode characters
                            if "\"" in all_text:
                                continue

                            all_results["task_name"].append(eval_task_name)
                            all_results["pred_final"].append(pred_final)
                            all_results["reasoning"].append(reasoning)
                            all_results["reference"].append(reference)

                            if "question" in all_results:
                                all_results["question"].append(question)
                            else:
                                all_results["question"] = [question]

                            save_item_cnt_total += 1
                            if save_item_cnt_total >= self.num_item_per_task:
                                finish_flag = True
                                break
                        else:
                            continue
                    case _:
                        raise ValueError(f">>> !!! >>> ValueError: task_type = {task_type}")

                if self.verbose and data_item_cnt_total % show_cnt == 0:
                    self.logger.info(f">>> Progress: [{ds_id}] [{data_item_cnt_total} / {len_dataset}]")

            # Done the current subtask/subset
            if self.verbose:
                self.logger.info(f">>> Done Subtask/Subset. [{ds_id}] "
                                 f"[data_item_cnt_ds = {data_item_cnt_ds}] "
                                 f"[save_item_cnt_total = {save_item_cnt_total}] "
                                 f"[end_with_eot_cnt_total = {end_with_eot_cnt_total}]\n")

        # Done all. Save all results to CSV
        if self.verbose:
            self.logger.info(f">>> Done ALL. [{eval_task_name}] "
                             f"[data_item_cnt_total = {data_item_cnt_total}] "
                             f"[save_item_cnt_total = {save_item_cnt_total}] "
                             f"[end_with_eot_cnt_total = {end_with_eot_cnt_total}]\n")

        return all_results


def main(
    task: int = 0,
    eval_task_name="ALL",
    hf_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    seed: int = 42,
    cuda: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
    output_dir: Optional[str] = None,
    num_item_per_task: int = 10,
    num_duplication: int = 1,
    num_item_in_a_row: int = 1,
    min_doc_length: int = 500,
    max_doc_length: int = 1000,
    **kwargs
) -> None:
    """
    Convert the result JSON files to CSV for Human evaluation.
    MATH_ALL: "gsm8k,math500"
    QA_ALL: "bbh,mmlu"
    SUM_ALL: "cnn_dailymail,xsum"

    :param task: 1. language model evaluation.
    :param eval_task_name: The name(s) of the evaluation task. (e.g., "xsum", "xlsum", and "gsm8k,math500")
    :param hf_id: ORGANIZATION_NAME/MODEL_NAME, e.g., "meta-llama/Llama-3.1-8B-Instruct"
    :param cache_dir: The root directory of the cache.
    :param project_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param verbose: Verbose mode: show logs.
    :param debug: Debugging / developing mode.
    :param output_dir: The path to the output file of the generated results.
    :param num_item_per_task: The number of items to use per task.
    :param num_duplication: The number duplications per item.
    :param num_item_in_a_row: The number of items in a CSV row.
    :param min_doc_length: The minimum summary length (the number of characters in the string).
    :param max_doc_length: The maximum summary length (the number of characters in the string).
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("Human_Eval_Intent")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Unused parameters in kwargs: {kwargs}")
    logger.info(f">>> cuda_dict: {cuda_dict}")

    if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
        os.environ["HF_HOME"] = cache_dir
        # os.environ["HF_HOME"] = os.path.join(cache_dir, "datasets")
        # os.environ["HF_HOME"] = os.path.join(cache_dir, "hub")
    else:
        cache_dir = None

    if eval_task_name == "MATH_ALL":
        eval_task_names = ["gsm8k", "math500"]
    elif eval_task_name == "QA_ALL":
        eval_task_names = ["ai2_arc", "bbh", "mmlu", "mmlu_pro"]
    elif eval_task_name == "SUM_ALL":
        eval_task_names = ["cnn_dailymail", "xsum", "xlsum", "samsum", "dialogsum", "wiki_lingua"]
    elif eval_task_name == "MATH_TWO":
        eval_task_names = ["gsm8k", "math500"]
    elif eval_task_name == "QA_TWO":
        eval_task_names = ["bbh", "mmlu"]
    elif eval_task_name == "SUM_TWO":
        eval_task_names = ["cnn_dailymail", "xsum"]
    else:
        eval_task_names = [eval_task_name]

    converter = HumanEvalIntent(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        eval_task_name=eval_task_name,
        cache_dir=cache_dir,
        project_dir=project_dir,
        hf_id=hf_id,
        debug=debug,
        output_dir=output_dir,
        num_item_per_task=num_item_per_task,
        min_doc_length=min_doc_length,
        max_doc_length=max_doc_length,
    )

    task = int(task)
    match task:
        case 1:
            # Preparing data for human evaluation
            all_task_res = dict()
            assert isinstance(eval_task_names, tuple) or isinstance(eval_task_names, list)
            for cur_task_name in eval_task_names:
                assert cur_task_name in converter.task_class_dict, \
                    f"AssertionError: task name {cur_task_name} not in task_class_dict"
                cur_task_name = str(cur_task_name).strip()
                logger.info(f">>> <START> {cur_task_name}\n")
                all_res = converter.do_convert(eval_task_name=cur_task_name)
                logger.info(f">>> <END> {cur_task_name}\n\n\n")

                # Put all results into `all_task_res`
                for _k, _v in all_res.items():
                    assert isinstance(_k, str) and isinstance(_v, list) and len(_v) > 0
                    _v = [json.dumps(str(__v).strip()) for __v in _v]
                    if _k not in all_task_res:
                        all_task_res[_k] = _v
                    else:
                        all_task_res[_k].extend(_v)

            # Validness check
            assert len(all_task_res) > 0
            num_items = -1
            for _k, _v in all_task_res.items():
                if num_items == -1:
                    num_items = len(_v)
                else:
                    assert num_items == len(_v)
            assert num_items > 0 and num_items % num_item_in_a_row == 0, \
                f"Assertion Error: num_items = {num_items}; num_item_in_a_row = {num_item_in_a_row}"

            # Group each `num_item_in_a_row` items in a CSV row (via JSON string)
            assert isinstance(num_duplication, int) and num_duplication >= 1
            all_res_group = dict()
            for _k, _v in all_task_res.items():
                for start_idx in range(0, len(_v), num_item_in_a_row):
                    cur_res_list = _v[start_idx: start_idx + num_item_in_a_row]
                    assert len(cur_res_list) == num_item_in_a_row

                    # Make the second item a dummy example for sanity check (not coherence)
                    # If the human worker give a high coherence rating on this item, we may not approve the evaluation
                    dummy_idx = 1
                    if _k == "reasoning":
                        raw_reasoning = cur_res_list[dummy_idx]
                        cur_intent_list = re.findall(r"<INTENT>.*?</INTENT>", raw_reasoning)
                        assert len(cur_intent_list) >= 2
                        new_reasoning = re.sub(r"<INTENT>.*?</INTENT>", "[TO_SPLIT]", raw_reasoning)
                        # random.shuffle(cur_intent_list)
                        # cur_intent_list.insert(0, cur_intent_list.pop())  # Shift right by 1 position
                        cur_intent_list = cur_intent_list[::-1]  # Reverse the intent list
                        for cur_intent in cur_intent_list:
                            new_reasoning = new_reasoning.replace("[TO_SPLIT]", cur_intent, 1)
                        assert "[TO_SPLIT]" not in new_reasoning
                        cur_res_list[dummy_idx] = new_reasoning

                    for _ in range(num_duplication):  # Let `num_duplication` different people evaluate the same item
                        for cur_idx, cur_res in enumerate(cur_res_list):
                            cur_key = f"{_k}-{cur_idx}"
                            if cur_key not in all_res_group:
                                all_res_group[cur_key] = [cur_res]
                            else:
                                all_res_group[cur_key].append(cur_res)

            assert len(all_res_group) > 0
            save_dir = os.path.join(project_dir, "human_eval")
            os.makedirs(save_dir, exist_ok=True)
            output_csv_fp = os.path.join(save_dir, f"swi_intent-{eval_task_name}-"
                                                   f"{num_item_per_task}_{num_duplication}_{num_item_in_a_row}.csv")
            logger.info(f"Results will be saved at: {output_csv_fp}")
            df_all_results = pd.DataFrame.from_dict(data=all_res_group, orient="columns")
            df_all_results.to_csv(output_csv_fp, index=False, header=True)
        case 2:
            # Statistics on the human evaluation results
            res_root_dir = os.path.join(project_dir, "results")

            csv_math = pd.read_csv(os.path.join(res_root_dir, "intent_math.csv"))
            csv_qa = pd.read_csv(os.path.join(res_root_dir, "intent_qa.csv"))
            csv_sum = pd.read_csv(os.path.join(res_root_dir, "intent_sum.csv"))

            res_math = [json.loads(str(_item).strip()) for _item in csv_math["Submitted Data"].tolist()]
            res_qa = [json.loads(str(_item).strip()) for _item in csv_qa["Submitted Data"].tolist()]
            res_sum = [json.loads(str(_item).strip()) for _item in csv_sum["Submitted Data"].tolist()]

            res_math = [_item["Data"]["taskData"][0] for _item in res_math]
            res_qa = [_item["Data"]["taskData"][0] for _item in res_qa]
            res_sum = [_item["Data"]["taskData"][0] for _item in res_sum]

            assert len(res_math) == len(res_qa) == len(res_sum) > 0
        case _:
            raise ValueError(f"ValueError: task = {task}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
