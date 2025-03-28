# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

from typing import Optional, Dict, Any

from tasks import EvalTaskManager


class EvalTaskLogiQA(EvalTaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_dir)

        # LogiQA: Question Answering (Multiple-Choice QA)
        # Train = 7376, Validation = 651, Test = 651
        # Features: ["label", "context", "question", "options"]
        # Eval: test set
        # >>> [use_swi = False] >>> #Sub-Tasks = 1; #Total Ins. = 651; avg_len_token: 185.476; std_len_token: 36.720
        # >>> [use_swi = True] >>> #Sub-Tasks = 1; #Total Ins. = 651; avg_len_token: 270.476; std_len_token: 36.720

        self.task_name = "logiqa"
        self.task_info = {
            "has_passage": True,
            "hf_dataset": [  # [hf_id, subset, eval_set]
                ["EleutherAI/logiqa", None, "test"],
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

        # Process data ["label", "context", "question", "options"]
        passage, question = str(data_item["context"]).strip(), str(data_item["question"]).strip()
        options, label = data_item["options"], str(data_item["label"])

        options = [str(_op).strip() for _op in options]
        label = label.strip().upper()
        label2index = {"A": 0, "B": 1, "C": 2, "D": 3}
        assert isinstance(options, list) and len(options) == 4 and label in label2index
        answer_str = options[label2index[label]]
        answer_label = label
        label_options = ["A", "B", "C", "D"]
        answer_options = options
        if not question.endswith("?"):
            question += "?"

        pos_a = label2index[label]
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
