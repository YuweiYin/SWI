# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
from typing import Optional, Dict, Any

from datasets import load_dataset

from tasks import EvalTaskManager


class EvalTaskMmlu(EvalTaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_dir)

        # MMLU: Question Answering (Multiple-Choice QA) (57 subtasks)
        # Train = 0, Validation = 1514, Test = 13842
        # Features: ["question", "subject", "choices", "answer"]
        # Eval: test set
        # >>> [use_swi = False] >>> #Sub-Tasks = 57; #Total Ins. = 14042; avg_len_token: 193.011; std_len_token: 91.976
        # >>> [use_swi = True] >>> #Sub-Tasks = 57; #Total Ins. = 14042; avg_len_token: 278.011; std_len_token: 91.976

        self.task_name = "mmlu"
        self.task_info = {
            "hf_dataset": [  # [hf_id, subset, eval_set]
                ["hails/mmlu_no_train", "formal_logic", "test"],  # Validation = 14, Test = 126
                ["hails/mmlu_no_train", "philosophy", "test"],  # Validation = 34, Test = 311
                ["hails/mmlu_no_train", "high_school_world_history", "test"],  # Validation = 26, Test = 237
                ["hails/mmlu_no_train", "international_law", "test"],  # Validation = 13, Test = 121
                ["hails/mmlu_no_train", "jurisprudence", "test"],  # Validation = 11, Test = 108
                ["hails/mmlu_no_train", "world_religions", "test"],  # Validation = 19, Test = 171
                ["hails/mmlu_no_train", "moral_disputes", "test"],  # Validation = 38, Test = 346
                ["hails/mmlu_no_train", "high_school_european_history", "test"],  # Validation = 18, Test = 165
                ["hails/mmlu_no_train", "logical_fallacies", "test"],  # Validation = 18, Test = 163
                ["hails/mmlu_no_train", "high_school_us_history", "test"],  # Validation = 22, Test = 204
                ["hails/mmlu_no_train", "moral_scenarios", "test"],  # Validation = 100, Test = 895
                ["hails/mmlu_no_train", "professional_law", "test"],  # Validation = 170, Test = 1534
                ["hails/mmlu_no_train", "prehistory", "test"],  # Validation = 35, Test = 324
                ["hails/mmlu_no_train", "us_foreign_policy", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "security_studies", "test"],  # Validation = 27, Test = 245
                ["hails/mmlu_no_train", "econometrics", "test"],  # Validation = 12, Test = 114
                ["hails/mmlu_no_train", "high_school_microeconomics", "test"],  # Validation = 26, Test = 238
                ["hails/mmlu_no_train", "sociology", "test"],  # Validation = 22, Test = 201
                ["hails/mmlu_no_train", "high_school_geography", "test"],  # Validation = 22, Test = 198
                ["hails/mmlu_no_train", "high_school_psychology", "test"],  # Validation = 60, Test = 545
                ["hails/mmlu_no_train", "professional_psychology", "test"],  # Validation = 69, Test = 612
                ["hails/mmlu_no_train", "high_school_macroeconomics", "test"],  # Validation = 43, Test = 390
                ["hails/mmlu_no_train", "high_school_government_and_politics", "test"],  # Validation = 21, Test = 193
                ["hails/mmlu_no_train", "public_relations", "test"],  # Validation = 12, Test = 110
                ["hails/mmlu_no_train", "human_sexuality", "test"],  # Validation = 12, Test = 131
                ["hails/mmlu_no_train", "miscellaneous", "test"],  # Validation = 86, Test = 783
                ["hails/mmlu_no_train", "medical_genetics", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "management", "test"],  # Validation = 11, Test = 103
                ["hails/mmlu_no_train", "virology", "test"],  # Validation = 18, Test = 166
                ["hails/mmlu_no_train", "nutrition", "test"],  # Validation = 33, Test = 306
                ["hails/mmlu_no_train", "global_facts", "test"],  # Validation = 10, Test = 100
                ["hails/mmlu_no_train", "marketing", "test"],  # Validation = 25, Test = 234
                ["hails/mmlu_no_train", "college_medicine", "test"],  # Validation = 22, Test = 173
                ["hails/mmlu_no_train", "clinical_knowledge", "test"],  # Validation = 29, Test = 265
                ["hails/mmlu_no_train", "professional_accounting", "test"],  # Validation = 31, Test = 282
                ["hails/mmlu_no_train", "professional_medicine", "test"],  # Validation = 31, Test = 272
                ["hails/mmlu_no_train", "human_aging", "test"],  # Validation = 23, Test = 223
                ["hails/mmlu_no_train", "business_ethics", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "college_physics", "test"],  # Validation = 11, Test = 102
                ["hails/mmlu_no_train", "elementary_mathematics", "test"],  # Validation = 41, Test = 378
                ["hails/mmlu_no_train", "machine_learning", "test"],  # Validation = 11, Test = 112
                ["hails/mmlu_no_train", "high_school_statistics", "test"],  # Validation = 23, Test = 216
                ["hails/mmlu_no_train", "electrical_engineering", "test"],  # Validation = 16, Test = 145
                ["hails/mmlu_no_train", "college_computer_science", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "anatomy", "test"],  # Validation = 14, Test = 135
                ["hails/mmlu_no_train", "high_school_physics", "test"],  # Validation = 17, Test = 151
                ["hails/mmlu_no_train", "high_school_computer_science", "test"],  # Valid = 9, Test = 100
                ["hails/mmlu_no_train", "computer_security", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "conceptual_physics", "test"],  # Validation = 26, Test = 235
                ["hails/mmlu_no_train", "college_mathematics", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "astronomy", "test"],  # Validation = 16, Test = 152
                ["hails/mmlu_no_train", "high_school_mathematics", "test"],  # Validation = 29, Test = 270
                ["hails/mmlu_no_train", "college_chemistry", "test"],  # Validation = 8, Test = 100
                ["hails/mmlu_no_train", "abstract_algebra", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "high_school_chemistry", "test"],  # Validation = 22, Test = 203
                ["hails/mmlu_no_train", "college_biology", "test"],  # Validation = 16, Test = 144
                ["hails/mmlu_no_train", "high_school_biology", "test"],  # Validation = 32, Test = 310
            ],
        }

        self.system_prompt_raw = f"""
You are a helpful assistant. \
You are good at answering questions and logical reasoning. \
You need to select one from the given options and answer A, B, C, or D.
Your final answer must start with "Final Answer:"
        """.strip()
        self.system_prompt_swi = f"""
You are a helpful assistant who speaks with intent. \
You are good at answering questions and logical reasoning. \
You need to select one from the given options and answer A, B, C, or D.
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
        for hf_ds in hf_ds_list:
            assert isinstance(hf_ds, list) and len(hf_ds) == 3
            # self.logger.info(f">>> [dataset: {hf_ds[0]} --- {hf_ds[1]}]")
            try:  # Load the subset
                cur_ds = load_dataset(
                    hf_ds[0],
                    hf_ds[1],
                    cache_dir=os.path.join(self.cache_dir, "datasets"),
                    trust_remote_code=True,
                )

                eval_split = hf_ds[2]
                assert eval_split in cur_ds
                assert "validation" in cur_ds and "test" in cur_ds
                len_train, len_valid, len_test = 0, len(cur_ds["validation"]), len(cur_ds["test"])
                assert len_valid > 0 and len_test > 0

                ds_dict = {
                    "hf_dataset": hf_ds[0],
                    "hf_subset": hf_ds[1],
                    "eval_split": eval_split,
                    "eval_dataset": cur_ds[eval_split],
                }

                self.logger.info(f">>> [dataset: {hf_ds[0]} --- {hf_ds[1]}] [eval_split = {eval_split}] "
                                 f"Train = {len_train}, Validation = {len_valid}, Test = {len_test}")

                dataset["data"].append(ds_dict)
            except Exception as e:
                if self.verbose:
                    self.logger.info(f">>> Exception: {e}")

        self.logger.info(f">>> [task_name: {self.task_name}] len(dataset) = {len(dataset['data'])}\n\n")  # 55
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

        # Process data
        question = str(data_item["question"]).strip()
        choices = list(data_item["choices"])
        choices = [str(_c).strip() for _c in choices]
        answer = int(data_item["answer"])

        assert isinstance(choices, list) and len(choices) == 4
        index2label = {0: "A", 1: "B", 2: "C", 3: "D"}
        assert answer in index2label

        answer_str = choices[answer]
        answer_label = index2label[answer]
        label_options = ["A", "B", "C", "D"]
        answer_options = choices

        pos_a = answer
        answers = [answer_str, answer_label, f"({answer_label})", f"{answer_label})"]

        # Set the main prompt (zero-shot)
        options_str = "\n".join([f"({_label}) {_ans}" for _label, _ans in zip(label_options, answer_options)])
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
                "options": answer_options,
                "ans_idx": pos_a,
            }
        }
        return result_dict
