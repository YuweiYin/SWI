#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import re
import time
from typing import Optional

import fire
import numpy as np

from transformers import AutoTokenizer
from datasets import Dataset

from tasks.tasks_utils import *
from utils.init_functions import logger_setup, random_setup
from utils.data_io import DataIO


def main(
    task: int = 1,
    cache_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    seed: int = 42,
    verbose: bool = False,
    hf_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    **kwargs
) -> None:
    """
    Statistics.

    :param task: 1. stat of input tokens; 2. stat of output tokens; 3. stat of intent verbs;
        4. aggregate the intent-verb stat by the task type; 5. confusion matrix between DA and SWI results.
    :param cache_dir: The root directory of the cache.
    :param project_dir: The directory of the project root.
    :param seed: Random seed of all modules.
    :param verbose: Verbose mode: show logs.
    :param hf_id: ORGANIZATION_NAME/MODEL_NAME, e.g., "meta-llama/Llama-3.1-8B-Instruct"
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setups
    logger = logger_setup("Intent_Statistics")
    random_setup(seed=seed, has_cuda=False)

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}\n")

    task = int(task)
    match task:
        case 1:
            # Stat of the input tokens (with or without SWI, i.e., extra tokens in the system prompt)
            all_sum = [TaskCnnDailymail, TaskXSum, TaskXlSum, TaskDialogSum, TaskWikiLingua]
            all_qa = [TaskBbh, TaskMmlu, TaskMmluPro]
            all_math = [TaskGSM8K, TaskGSM8KPlatinum, TaskMATH500]
            for eval_class in all_sum + all_qa + all_math:
                eval_tasks = eval_class(verbose=verbose, logger=logger, cache_dir=cache_dir, project_dir=project_dir)
                eval_tasks.token_stat(eval_task_obj=eval_tasks, cache_dir=cache_dir, hf_id=hf_id, use_swi=False)
                eval_tasks.token_stat(eval_task_obj=eval_tasks, cache_dir=cache_dir, hf_id=hf_id, use_swi=True)
        case 2:
            # Stat of the output tokens (LLMs' generation using or not using SWI)
            assert "output_filepath" in kwargs, kwargs
            output_filepath = kwargs["output_filepath"]
            assert isinstance(output_filepath, str) and os.path.isfile(output_filepath), output_filepath

            if output_filepath.endswith(".json"):
                outputs = DataIO.load_json(output_filepath, verbose=verbose)
            else:
                raise ValueError(f">>> !!! >>> The output file must be JSON: {output_filepath}")
            assert isinstance(outputs, dict) and len(outputs) > 0, type(outputs)

            # Set the tokenizer path
            os.environ["HF_HOME"] = cache_dir
            hf_name = "--".join(hf_id.split("/"))
            local_model_path = os.path.join(cache_dir, "models--" + hf_name, "snapshots/model")
            model_path = local_model_path if os.path.isdir(local_model_path) else hf_id

            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, padding_side="left", truncation_side="left", cache_dir=cache_dir,
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

            all_stat = dict()
            all_num_token = []
            for ds_id, cur_results in outputs.items():
                assert isinstance(cur_results, list) and len(cur_results) > 0, type(cur_results)
                if verbose:
                    logger.info(f">>> [Dataset: {ds_id}] [#Items = {len(cur_results)}]")

                cur_stat = []
                for cur_gen_output in cur_results:
                    assert isinstance(cur_gen_output, dict) and len(cur_gen_output) > 0, type(cur_gen_output)
                    output_text = cur_gen_output["output_text"]

                    assert isinstance(output_text, str)
                    tokenized_ids = tokenizer(
                        output_text,
                        padding=False,  # truncation=True, max_length=1024
                        return_tensors="pt",
                    )
                    num_token = tokenized_ids.data["input_ids"].size(-1)

                    cur_stat.append({
                        "output_text": output_text,
                        "num_char": len(output_text),
                        "num_token": num_token,
                    })
                    all_num_token.append(num_token)

                all_stat[ds_id] = cur_stat

            # Show stat logs
            assert len(all_num_token) > 0
            # avg_num_token = sum(all_num_token) / len(all_num_token)
            avg_num_token = float(np.mean(all_num_token))
            std_num_token = float(np.std(all_num_token))
            logger.info(
                f">>> DONE ALL. hf_id = {hf_id}\n"
                f">>> output_filepath = {output_filepath}\n"
                f">>> #Sub-Tasks = {len(all_stat)}; #Total Ins. = {len(all_num_token)}; "
                f"avg_num_token: {avg_num_token:.3f}; std_num_token: {std_num_token:.3f}\n\n"
            )
        case 3:
            # Stat of the intents (i.e., to count the verbs in the specified intent format: "To do something")
            assert "output_filepath" in kwargs, kwargs
            output_filepath = kwargs["output_filepath"]
            assert isinstance(output_filepath, str) and os.path.isfile(output_filepath), output_filepath

            if output_filepath.endswith(".json"):
                outputs = DataIO.load_json(output_filepath, verbose=verbose)
            else:
                raise ValueError(f">>> !!! >>> The output file must be JSON: {output_filepath}")
            assert isinstance(outputs, dict) and len(outputs) > 0, type(outputs)

            intent_ptn = re.compile(r"<INTENT>(.*?)</INTENT>")

            total_instances = 0
            total_verbs = 0
            all_intents = dict()
            for ds_id, cur_results in outputs.items():
                assert isinstance(cur_results, list) and len(cur_results) > 0, type(cur_results)
                if verbose:
                    logger.info(f">>> [Dataset: {ds_id}] [#Items = {len(cur_results)}]")
                total_instances += len(cur_results)

                for cur_gen_output in cur_results:
                    assert isinstance(cur_gen_output, dict) and len(cur_gen_output) > 0, type(cur_gen_output)
                    output_text = cur_gen_output["output_text"]

                    assert isinstance(output_text, str)
                    cur_intents = re.findall(intent_ptn, output_text)
                    if isinstance(cur_intents, list) and len(cur_intents) > 0:
                        # Extract the verb
                        for cur_intent in cur_intents:
                            cur_intent_raw = str(cur_intent).strip()
                            cur_intent_lower = cur_intent_raw.lower()
                            if len(cur_intent_lower) > 0 and cur_intent_lower.startswith("to "):
                                cur_verb = cur_intent_lower.lstrip("to ").strip().split()[0]
                                if cur_verb not in all_intents:
                                    all_intents[cur_verb] = {"count": 1, "intent": [cur_intent_raw]}
                                else:
                                    all_intents[cur_verb]["count"] += 1
                                    all_intents[cur_verb]["intent"].append(cur_intent_raw)
                                total_verbs += 1

            # Show stat logs and save the results (sorted by the frequency of the verbs)
            if len(all_intents) == 0:
                logger.info(f">>> !!! >>> len(all_intents) == 0; output_filepath = {output_filepath}\nExit")
            else:
                all_verbs = list(all_intents.keys())
                sorted_verbs = sorted(all_verbs, key=lambda x: all_intents[x]["count"], reverse=True)
                sorted_intents = [
                    {"verb": _verb, "count": all_intents[_verb]["count"], "intent": all_intents[_verb]["intent"]}
                    for _verb in sorted_verbs
                ]
                output_dir = os.path.dirname(output_filepath)
                assert os.path.isdir(output_dir), output_dir
                save_filepath = os.path.join(output_dir, "intent_stat.json")
                DataIO.save_json(save_filepath, sorted_intents)

                show_num = max(1, min(20, len(sorted_intents)))
                logger.info(
                    f">>> DONE ALL. hf_id = {hf_id}; output_filepath = {output_filepath}\n"
                    f">>> save_filepath: {save_filepath}\n"
                    f">>> # of total verbs: {total_verbs}\n"
                    f">>> # of unique verbs: {len(set(all_verbs))}\n"
                    f">>> # of verbs per instance: {total_verbs / total_instances} [#Ins. = {total_instances}]\n"
                    f">>> Top {show_num} verbs:"
                )
                for intent_dict in sorted_intents[:show_num]:
                    cur_verb, cur_cnt = intent_dict["verb"], intent_dict["count"]
                    logger.info(f">>> >>> {cur_verb} [count = {cur_cnt}] [ratio = {cur_cnt / total_verbs}]")
        case 4:
            # Aggregate the intent-verb stat by the task type
            assert "output_dir" in kwargs, kwargs
            output_dir = kwargs["output_dir"]
            assert isinstance(output_dir, str) and os.path.isdir(output_dir), output_dir

            assert "stat_task_type" in kwargs, kwargs
            stat_task_type = kwargs["stat_task_type"]
            assert isinstance(stat_task_type, str) and stat_task_type in {"sum", "qa", "math"}, stat_task_type
            if stat_task_type == "sum":
                dataset_names = list(SUM_CLASS_DICT.keys())
            elif stat_task_type == "qa":
                dataset_names = list(QA_CLASS_DICT.keys())
            else:
                assert stat_task_type == "math", stat_task_type
                dataset_names = list(MATH_CLASS_DICT.keys())

            all_intents = dict()
            total_verbs = 0
            for ds_name in dataset_names:
                hf_name = "--".join(hf_id.split("/"))
                cur_intent_fp = os.path.join(output_dir, ds_name, hf_name, "intent_stat.json")
                assert os.path.isfile(cur_intent_fp), cur_intent_fp
                cur_sorted_intents = DataIO.load_json(cur_intent_fp)
                for cur_intent in cur_sorted_intents:
                    cur_verb = str(cur_intent["verb"])
                    cur_count = int(cur_intent["count"])
                    cur_intent_statements = list(cur_intent["intent"])
                    if cur_verb not in all_intents:
                        all_intents[cur_verb] = {
                            "verb": cur_verb, "count": cur_count, "intent": cur_intent_statements,
                        }
                    else:
                        all_intents[cur_verb]["count"] += cur_count
                        all_intents[cur_verb]["intent"].extend(cur_intent_statements)
                    total_verbs += cur_count

            if len(all_intents) == 0:
                logger.info(f">>> !!! >>> len(all_intents) == 0; output_dir = {output_dir}\nExit")
            else:
                all_verbs = list(all_intents.keys())
                sorted_verbs = sorted(all_verbs, key=lambda x: all_intents[x]["count"], reverse=True)
                sorted_intents = [
                    {"verb": _verb, "count": all_intents[_verb]["count"], "intent": all_intents[_verb]["intent"]}
                    for _verb in sorted_verbs
                ]
                assert os.path.isdir(output_dir), output_dir
                save_filepath = os.path.join(output_dir, f"intent_stat-{stat_task_type}.json")
                DataIO.save_json(save_filepath, sorted_intents)

                show_num = max(1, min(20, len(sorted_intents)))
                logger.info(
                    f">>> DONE ALL. hf_id = {hf_id}; output_dir = {output_dir}\n"
                    f">>> save_filepath: {save_filepath}\n"
                    f">>> # of total verbs: {total_verbs}\n"
                    f">>> # of unique verbs: {len(set(all_verbs))}\n"
                    # f">>> # of verbs per instance: {total_verbs / total_instances} [#Ins. = {total_instances}]\n"
                    f">>> Top {show_num} verbs:"
                )
                for intent_dict in sorted_intents[:show_num]:
                    cur_verb, cur_cnt = intent_dict["verb"], intent_dict["count"]
                    logger.info(f">>> >>> {cur_verb} [count = {cur_cnt}] [ratio = {cur_cnt / total_verbs}]")
        case 5:
            # 5. Count the number of correct instances for the evaluation results (using or not using SWI)
            assert "eval_filepath_da" in kwargs and "eval_filepath_swi" in kwargs, kwargs
            eval_filepath_da, eval_filepath_swi = kwargs["eval_filepath_da"], kwargs["eval_filepath_swi"]
            assert isinstance(eval_filepath_da, str) and os.path.isfile(eval_filepath_da), eval_filepath_da
            assert isinstance(eval_filepath_swi, str) and os.path.isfile(eval_filepath_swi), eval_filepath_swi

            assert eval_filepath_da.endswith(".json")
            eval_results_da = DataIO.load_json(eval_filepath_da, verbose=verbose)
            assert isinstance(eval_results_da, dict) and len(eval_results_da) > 0, type(eval_results_da)

            assert eval_filepath_swi.endswith(".json")
            eval_results_swi = DataIO.load_json(eval_filepath_swi, verbose=verbose)
            assert isinstance(eval_results_swi, dict) and len(eval_results_swi) > 0, type(eval_results_swi)

            assert "eval_task_name" in kwargs, kwargs
            eval_task_name = str(kwargs["eval_task_name"])
            assert eval_task_name in TASK_CLASS_DICT, f"AssertionError: task {eval_task_name} not in TASK_CLASS_DICT"
            eval_task_class = TASK_CLASS_DICT[eval_task_name]
            eval_task_obj = eval_task_class(
                verbose=verbose,
                logger=logger,
                cache_dir=cache_dir,
                project_dir=project_dir,
            )
            logger.info(f">>> Evaluation Task: {eval_task_name}")
            task_info = eval_task_obj.load_task()
            dataset_list = task_info["data"]

            assert "eval_metric" in kwargs, kwargs
            eval_metric = str(kwargs["eval_metric"])

            all_stat = dict()
            both_correct_cnt, both_incorrect_cnt, only_da_correct_cnt, only_swi_correct_cnt = 0, 0, 0, 0
            for dataset_dict in dataset_list:
                ds_name, subset = dataset_dict["hf_dataset"], dataset_dict["hf_subset"]
                eval_split, eval_dataset = dataset_dict["eval_split"], dataset_dict["eval_dataset"]
                assert isinstance(eval_dataset, Dataset)
                len_dataset = len(eval_dataset)
                assert isinstance(ds_name, str) and len(ds_name) > 0
                if isinstance(subset, str) and len(subset) > 0:
                    ds_id = f"{ds_name}---{subset}"
                else:
                    ds_id = ds_name
                logger.info(f">>> [Dataset: {ds_id}] [Eval: {eval_split}] # = {len_dataset}")

                assert ds_id in eval_results_da and ds_id in eval_results_swi
                cur_results_da, cur_results_swi = eval_results_da[ds_id], eval_results_swi[ds_id]
                assert isinstance(cur_results_da, dict) and isinstance(cur_results_swi, dict)
                assert "ds_results" in cur_results_da and "ds_results" in cur_results_swi
                cur_results_da, cur_results_swi = cur_results_da["ds_results"], cur_results_swi["ds_results"]
                assert isinstance(cur_results_da, list) and isinstance(cur_results_swi, list)
                assert len(cur_results_da) == len(cur_results_swi) == len_dataset > 0

                cur_stat = {
                    "both_correct": [],
                    "both_incorrect": [],
                    "only_da_correct": [],
                    "only_swi_correct": [],
                }
                for res_da, res_swi in zip(cur_results_da, cur_results_swi):
                    eval_score_da = res_da["eval_score"]
                    if eval_metric in eval_score_da and isinstance(eval_score_da[eval_metric], dict):
                        eval_score_da = eval_score_da[eval_metric]["score"]
                        assert eval_score_da == 1.0 or eval_score_da == 0.0, eval_score_da
                    else:
                        eval_score_da = 0.0

                    eval_score_swi = res_swi["eval_score"]
                    if eval_metric in eval_score_swi and isinstance(eval_score_swi[eval_metric], dict):
                        eval_score_swi = eval_score_swi[eval_metric]["score"]
                        assert eval_score_swi == 1.0 or eval_score_swi == 0.0, eval_score_swi
                    else:
                        eval_score_swi = 0.0

                    if eval_score_da == 1.0 and eval_score_swi == 1.0:
                        cur_stat["both_correct"].append({"da": res_da, "swi": res_swi})
                        both_correct_cnt += 1
                    elif eval_score_da == 1.0 and eval_score_swi == 0.0:
                        cur_stat["only_da_correct"].append({"da": res_da, "swi": res_swi})
                        only_da_correct_cnt += 1
                    elif eval_score_da == 0.0 and eval_score_swi == 1.0:
                        cur_stat["only_swi_correct"].append({"da": res_da, "swi": res_swi})
                        only_swi_correct_cnt += 1
                    elif eval_score_da == 0.0 and eval_score_swi == 0.0:
                        cur_stat["both_incorrect"].append({"da": res_da, "swi": res_swi})
                        both_incorrect_cnt += 1
                    else:
                        raise ValueError(f">>> ValueError: eval_score_da ({eval_score_da}) and "
                                         f"eval_score_swi ({eval_score_swi}) must be either 0.0 or 1.0!")

                all_stat[ds_id] = cur_stat
                logger.info(
                    f">>> DONE [ds_id = {ds_id}] "
                    f"both_correct = {len(cur_stat['both_correct'])}; "
                    f"only_da_correct = {len(cur_stat['only_da_correct'])}; "
                    f"only_swi_correct = {len(cur_stat['only_swi_correct'])}; "
                    f"both_incorrect = {len(cur_stat['both_incorrect'])}"
                )

            # Show stat logs
            logger.info(
                f">>> DONE ALL [Task: {eval_task_name}] "
                f"both_correct = {both_correct_cnt}; "
                f"only_da_correct = {only_da_correct_cnt}; "
                f"only_swi_correct = {only_swi_correct_cnt}; "
                f"both_incorrect = {both_incorrect_cnt}"
            )
        case _:
            raise ValueError(f"ValueError: task = {task}")

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
