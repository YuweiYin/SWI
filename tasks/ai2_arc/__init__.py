# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

from typing import Optional, Dict, Any

from tasks import EvalTaskManager


class EvalTaskAi2Arc(EvalTaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_dir)

        # AI2 ARC: Question Answering (Multiple-Choice QA)
        # Train = 3370, Valid = 869, Test = 3548
        # Features: ["id", "question", "choices", "answerKey"]
        # Eval: test set
        # >>> [use_swi = False] >>> #Sub-Tasks = 2; #Total Ins. = 3548; avg_len_token: 144.014; std_len_token: 21.760
        # >>> [use_swi = True] >>> #Sub-Tasks = 2; #Total Ins. = 3548; avg_len_token: 229.014; std_len_token: 21.760

        self.task_name = "ai2_arc"
        self.task_info = {
            "hf_dataset": [  # [hf_id, subset, eval_set]
                ["allenai/ai2_arc", "ARC-Easy", "test"],  # Train = 2251, Validation = 570, Test = 2376
                ["allenai/ai2_arc", "ARC-Challenge", "test"],  # Train = 1119, Validation = 299, Test = 1172
            ],
        }

        self.system_prompt_raw = f"""
You are a helpful assistant. \
You are good at answering questions and logical reasoning. \
You need to select one from the given options and answer A, B, C, etc.
Your final answer must start with "Final Answer:"
        """.strip()
        self.system_prompt_swi = f"""
You are a helpful assistant who speaks with intent. \
You are good at answering questions and logical reasoning. \
You need to select one from the given options and answer A, B, C, etc.
Your final answer must start with "Final Answer:"
During generation, follow all the requirements below:
1. Always explicitly state your own intent before speaking each sentence.
2. Each intent statement should explain the sentence followed up.
3. Your intent must start with the "<INTENT>" tag and end with the "</INTENT>" tag.
4. At last, clearly and concisely give your final answer starting with "Final Answer:"
        """.strip()

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

        # Process data ["id", "question", "choices", "answerKey"]
        question = str(data_item["question"]).strip()
        options = data_item["choices"]
        answer = str(data_item["answerKey"]).strip()

        assert isinstance(options, dict) and "label" in options and "text" in options
        label_options = options["label"]
        answer_options = options["text"]
        answer_options = [str(_op).strip() for _op in answer_options]
        assert isinstance(label_options, list)  # and len(label_options) == 3/4/5
        assert isinstance(answer_options, list) and len(answer_options) == len(label_options)

        label2index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        assert answer in label2index, f"Assertion Error: answer not in label2index: answer = {answer}"
        answer_label = answer
        answer_str = answer_options[label2index[answer]]
        pos_a = label2index[answer]
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
