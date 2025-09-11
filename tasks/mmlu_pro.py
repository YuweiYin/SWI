# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

from typing import Optional, Dict, Any

from tasks import TaskManager


class TaskMmluPro(TaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(verbose, logger, cache_dir, project_dir, **kwargs)

        # MMLU-Pro: Question Answering (Multiple-Choice QA)
        # Train = 0, Validation = 70, Test = 12032

        self.task_name = "mmlu_pro"
        self.task_info = {
            "hf_dataset": [  # [hf_id, subset, eval_set]
                ["TIGER-Lab/MMLU-Pro", None, "test"],
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
2. Each intent statement should explain the sentence that follows.
3. Your intent must start with the "<INTENT>" tag and end with the "</INTENT>" tag. \
The content within the intent tags must begin with "To" followed by a verb, such as "To accomplish a task."
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
        options = list(data_item["options"])
        options = [str(_op).strip() for _op in options]
        # answer = str(data_item["answer"]).strip()
        answer_index = int(data_item["answer_index"])

        assert isinstance(options, list)
        index2label = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J"}
        # assert (answer in label2index) and (answer_index == label2index[answer])
        assert answer_index in index2label
        answer_str = options[answer_index]
        answer_label = index2label[answer_index]
        label_options = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        answer_options = options

        pos_a = answer_index
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
