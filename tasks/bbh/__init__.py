# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import re
import json
from typing import Optional, Dict, Any

from datasets import load_dataset

from tasks import EvalTaskManager


class EvalTaskBbh(EvalTaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_dir)

        # BBH: Question Answering (27 subtasks: 23 Multiple-Choice QA + 4 Open QA)
        # Train = 0, Valid = 0, Test = 6511 (only MCQA: 5511)
        # Features: ["input", "target"]
        # Eval: test set
        # >>> [use_swi = False] >>> #Sub-Tasks = 23; #Total Ins. = 5511; avg_len_token: 202.500; std_len_token: 73.391
        # >>> [use_swi = True] >>> #Sub-Tasks = 23; #Total Ins. = 5511; avg_len_token: 287.500; std_len_token: 73.391

        self.task_name = "bbh"
        self.task_info = {
            "hf_dataset": [  # [hf_id, subset, eval_set]
                ["lukaemon/bbh", "word_sorting", "test"],  # Test = 250 (not MCQA)
                ["lukaemon/bbh", "web_of_lies", "test"],  # Test = 250
                ["lukaemon/bbh", "tracking_shuffled_objects_three_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "tracking_shuffled_objects_seven_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "tracking_shuffled_objects_five_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "temporal_sequences", "test"],  # Test = 250
                ["lukaemon/bbh", "sports_understanding", "test"],  # Test = 250
                ["lukaemon/bbh", "snarks", "test"],  # Test = 178
                ["lukaemon/bbh", "salient_translation_error_detection", "test"],  # Test = 250
                ["lukaemon/bbh", "ruin_names", "test"],  # Test = 250
                ["lukaemon/bbh", "reasoning_about_colored_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "penguins_in_a_table", "test"],  # Test = 146
                ["lukaemon/bbh", "object_counting", "test"],  # Test = 250 (not MCQA)
                ["lukaemon/bbh", "navigate", "test"],  # Test = 250
                ["lukaemon/bbh", "multistep_arithmetic_two", "test"],  # Test = 250 (not MCQA)
                ["lukaemon/bbh", "movie_recommendation", "test"],  # Test = 250
                ["lukaemon/bbh", "logical_deduction_three_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "logical_deduction_seven_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "logical_deduction_five_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "hyperbaton", "test"],  # Test = 250
                ["lukaemon/bbh", "geometric_shapes", "test"],  # Test = 250
                ["lukaemon/bbh", "formal_fallacies", "test"],  # Test = 250
                ["lukaemon/bbh", "dyck_languages", "test"],  # Test = 250 (not MCQA)
                ["lukaemon/bbh", "disambiguation_qa", "test"],  # Test = 250
                ["lukaemon/bbh", "date_understanding", "test"],  # Test = 250
                ["lukaemon/bbh", "causal_judgement", "test"],  # Test = 187
                ["lukaemon/bbh", "boolean_expressions", "test"],  # Test = 250
            ],
        }
        self.open_qa_subtasks = {"word_sorting", "object_counting", "dyck_languages", "multistep_arithmetic_two"}

        self.options_fp = os.path.join(project_dir, "tasks", self.task_name, "options.json")
        assert os.path.isfile(self.options_fp)
        with open(self.options_fp, "r", encoding="utf-8") as fp_in:
            self.options = json.load(fp_in)

        self.system_prompt_raw = f"""
You are a helpful assistant. \
You are good at answering questions and logical reasoning. \
For multiple-choice questions, you need to select one from the given options.
Your final answer must start with "Final Answer:"
        """.strip()
        self.system_prompt_swi = f"""
You are a helpful assistant who speaks with intent. \
You are good at answering questions and logical reasoning. \
For multiple-choice questions, you need to select one from the given options.
Your final answer must start with "Final Answer:"
During generation, follow all the requirements below:
1. Always explicitly state your own intent before speaking each sentence.
2. Each intent statement should explain the sentence followed up.
3. Your intent must start with the "<INTENT>" tag and end with the "</INTENT>" tag.
4. At last, clearly and concisely give your final answer starting with "Final Answer:"
        """.strip()

    def load_task(
            self,
    ) -> Dict[str, Any]:
        assert isinstance(self.task_name, str) and self.task_name in self.all_tasks
        assert isinstance(self.task_info, dict)
        hf_ds_list = self.task_info["hf_dataset"]
        assert isinstance(hf_ds_list, list) and len(hf_ds_list) > 0

        self.logger.info(f">>> [task_name: {self.task_name}]")
        dataset = {
            "task_name": self.task_name,
            "data": [],
        }

        option_pattern = re.compile(r"^\([A-Z]\)$")
        for hf_ds in hf_ds_list:
            assert isinstance(hf_ds, list) and len(hf_ds) == 3
            if hf_ds[1] in self.open_qa_subtasks:  # ignore subtasks that are not multi-choice QA
                self.logger.info(f">>> Ignore subtask: [dataset: {hf_ds[0]} --- {hf_ds[1]}]")
                continue

            try:  # Load the subset
                cur_ds = load_dataset(
                    hf_ds[0],
                    hf_ds[1],
                    cache_dir=os.path.join(self.cache_dir, "datasets"),
                    trust_remote_code=True,
                )

                eval_split = hf_ds[2]
                assert eval_split in cur_ds

                assert "test" in cur_ds
                len_train, len_valid, len_test = 0, 0, len(cur_ds["test"])
                assert len_test > 0
                eval_dataset = cur_ds["test"]

                if hf_ds[1] not in self.open_qa_subtasks:
                    # For multiple-choice QA subtasks, obtain all possible options ("target")
                    options = list(set(list(eval_dataset["target"])))
                    options = [str(_op).strip() for _op in options]
                    # ensure the format of option labels (other: Yes/No, True/False, valid/invalid)
                    # Note: this leads to certain evaluation acc=0 (can not match target and option)
                    if "(A)" in options:
                        options = [_op for _op in options if re.match(option_pattern, _op) is not None]
                    options.sort()
                else:
                    options = None

                ds_dict = {
                    "hf_dataset": hf_ds[0],
                    "hf_subset": hf_ds[1],
                    "eval_split": eval_split,
                    "eval_dataset": eval_dataset,
                    "options": options,
                }

                self.logger.info(f">>> [dataset: {hf_ds[0]} --- {hf_ds[1]}] [eval_split = {eval_split}] "
                                 f"Train = {len_train}, Validation = {len_valid}, Test = {len_test}")

                dataset["data"].append(ds_dict)
            except Exception as e:
                if self.verbose:
                    self.logger.info(f">>> Exception: {e}")

        self.logger.info(f">>> [task_name: {self.task_name}] len(dataset) = {len(dataset['data'])}\n\n")  # 27
        return dataset

    def set_dialog(
            self,
            ds_name: str,
            subset: str,
            data_item,
            use_swi: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        assert isinstance(self.task_name, str) and self.task_name in self.all_tasks
        use_cot = "use_cot" in kwargs and kwargs["use_cot"]
        use_arr = "use_arr" in kwargs and kwargs["use_arr"]
        use_ps = "use_ps" in kwargs and kwargs["use_ps"]

        # Load data
        hf_ds_list = self.task_info["hf_dataset"]
        assert isinstance(hf_ds_list, list) and len(hf_ds_list) > 0

        dialog_sys = []
        if use_swi:
            dialog_sys.append({"role": "system", "content": self.system_prompt_swi})
        else:
            dialog_sys.append({"role": "system", "content": self.system_prompt_raw})

        # Process data ["input", "target"]
        question = str(data_item["input"]).strip()
        answer = str(data_item["target"]).strip()
        answers = [answer]

        # Get all options for the "target"
        if subset in self.options and answer in list(self.options[subset]):  # Multiple-choice QA subtasks
            options = list(self.options[subset])
            options.sort()
            pos_a = options.index(answer)

            # Set the main prompt (zero-shot)
            options_str = "\n".join([f"{_op}" for _op in options])
            if use_swi:  # SWI (ours): Speaking with Intent
                dialog_user = [{"role": "user", "content": f"""
Speak with intent and answer the following question by selecting an option.\n
{question}
{options_str}
                """.strip()}]
            else:  # Baseline: LLM Generation without Intent
                dialog_user = [{"role": "user", "content": f"""
Answer the following question by selecting an option.\n
{question}
{options_str}
                """.strip()}]
        else:  # Not a Multiple-choice QA subtask (ignored)
            options = None
            pos_a = None
            if use_swi:  # SWI (ours): Speaking with Intent
                dialog_user = [{"role": "user", "content": f"""
Speak with intent and answer the following question.\n
{question}
                """.strip()}]
            else:  # Baseline: LLM Generation without Intent
                dialog_user = [{"role": "user", "content": f"""
Answer the following question.\n
{question}
                """.strip()}]

        # Answer Trigger Prompting methods
        if use_cot:  # Zero-shot Chain-of-Thought (CoT) prompting  https://arxiv.org/abs/2205.11916
            dialog_user[0]["content"] = dialog_user[0]["content"] + "\n\n" + f"""
Let's think step by step.
            """.strip()
        elif use_arr:  # ARR: Analyzing, Retrieving, and Reasoning  https://arxiv.org/abs/2502.04689
            dialog_user[0]["content"] = dialog_user[0]["content"] + "\n\n" + f"""
Let's analyze the intent of the question, find relevant information, \
and answer the question with step-by-step reasoning.
            """.strip()
        elif use_ps:  # Plan-and-Solve prompting  https://aclanthology.org/2023.acl-long.147/
            dialog_user[0]["content"] = dialog_user[0]["content"] + "\n\n" + f"""
Let's first understand the problem and devise a plan to solve the problem. \
Then, let's carry out the plan and solve the problem step by step.
            """.strip()
        else:  # No extra prompts
            pass

        dialog = dialog_sys + dialog_user
        # prompt = self.tokenizer.apply_chat_template(
        #     dialog,
        #     tokenize=False,
        #     padding=False,
        #     add_generation_prompt=True,
        #     return_tensors=None
        # )

        # Set the result dict
        result_dict = {
            "dialog": dialog,
            "answers": answers,
            "info": {
                "task_type": "qa",
                "context": "",
                "question": question,
                "options": options,
                "ans_idx": pos_a,
            }
        }
        return result_dict
