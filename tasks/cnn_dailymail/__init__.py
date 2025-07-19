# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

from typing import Optional, Dict, Any

from tasks import EvalTaskManager


class EvalTaskCnnDailymail(EvalTaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(verbose, logger, cache_dir, project_dir)

        # CNN / DailyMail: Summarization
        # Train = 287113, Valid = 13368, Test = 11490
        # Features: ["article", "highlights", "id"]
        # Eval: test set
        # >>> [use_swi = False] >>> #Sub-Tasks = 1; #Total Ins. = 11490; avg_len_token: 920.255; std_len_token: 438.472
        # >>> [use_swi = True] >>> #Sub-Tasks = 1; #Total Ins. = 11490; avg_len_token: 1063.255; std_len_token: 438.472

        self.task_name = "cnn_dailymail"
        self.task_info = {
            "hf_dataset": [  # [hf_id, subset, eval_set]
                ["abisee/cnn_dailymail", "3.0.0", "test"],
            ],
        }

        add_def = "add_def" in kwargs and kwargs["add_def"]
        intent_def = """
The intent is a usually clearly formulated or planned intention, or the act or fact of intending. \
Some synonyms of intent are intention, purpose, aim, goal, and objective.
        """.strip()

        self.system_prompt_raw = """
You are a helpful assistant. \
You are good at summarizing documents and the summary must start with "Final Summary:"
        """.strip()
        self.system_prompt_swi = """
You are a helpful assistant who speaks with intent. \
You are good at summarizing documents and the summary must start with "Final Summary:"
During generation, follow all the requirements below:
1. Always explicitly state your own intent before speaking each sentence.
2. Each intent statement should explain the sentence that follows.
3. Your intent must start with the "<INTENT>" tag and end with the "</INTENT>" tag. \
The content within the intent tags must begin with "To" followed by a verb, such as "To accomplish a task."
4. At last, clearly and concisely give your final summary starting with "Final Summary:"
        """.strip()

        # SWI prompt variants
        self.system_prompt_swi_v1 = """
You are a purposeful assistant skilled in document summarization who speaks with intent. \
Your final response must begin with "Final Summary:"
While generating responses, adhere strictly to these instructions:
1. Before every sentence, clearly state your intent using an explanation.
2. Each intention should directly clarify the sentence that follows.
3. Use the tags "<INTENT>" and "</INTENT>" to wrap each intent statement. \
Each statement inside the intent tags must begin with "To" and a verb, for example, "To describe the process."
4. Conclude with a clear and concise final summary that begins with "Final Summary:"
        """.strip()
        self.system_prompt_swi_v2 = """
You are a helpful assistant who is skilled in text summarization and always communicates with deliberate intent. \
Ensure your final output starts with "Final Summary:"
Comply with the following instructions during your response:
1. Begin each sentence with a description of your intent.
2. The intent must directly relate to and explain the sentence that comes after it.
3. Surround each intent with the tags "<INTENT>" and "</INTENT>". \
Each intent statement enclosed by the tags should start with the word "To" and an action verb, \
like "To explain the reasoning."
4. Finish with a succinct summary, introduced by "Final Summary:"
        """.strip()
        self.system_prompt_swi_v3 = """
You are a precise and helpful assistant proficient in text summarization, who always speaks with deliberate intent. \
Your final response must begin with "Final Summary:"
While producing your response, follow these guidelines:
1. Before each sentence, declare your intent explicitly.
2. Ensure each intent explains the sentence that immediately follows.
3. Wrap every intent declaration with "<INTENT>" and "</INTENT>" tags. \
Make sure that every intent statement within the tags begins with "To" and an action verb, \
for example, "To justify the choice."
4. Conclude your response with a clearly stated final summary prefaced by "Final Summary:"
        """.strip()

        self.system_prompt_swi_all = [
            self.system_prompt_swi, self.system_prompt_swi_v1, self.system_prompt_swi_v2, self.system_prompt_swi_v3]
        if add_def:
            self.system_prompt_swi_all = [intent_def + "\n\n" + _p for _p in self.system_prompt_swi_all]

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

        if "swi_version" in kwargs and isinstance(kwargs["swi_version"], int):
            system_prompt_swi_idx = kwargs["swi_version"]
        else:
            system_prompt_swi_idx = 0
        assert 0 <= system_prompt_swi_idx < len(self.system_prompt_swi_all)

        # Load data
        hf_ds_list = self.task_info["hf_dataset"]
        assert isinstance(hf_ds_list, list) and len(hf_ds_list) > 0

        dialog_sys = []
        if use_swi:
            dialog_sys.append({"role": "system", "content": self.system_prompt_swi_all[system_prompt_swi_idx]})
        else:
            dialog_sys.append({"role": "system", "content": self.system_prompt_raw})

        # Process data ["article", "highlights", "id"]
        article = str(data_item["article"]).strip().replace("\n\n", "\n")
        summary = str(data_item["highlights"]).strip()
        answers = [summary]

        # Set the main prompt (zero-shot)
        if use_swi:  # SWI (ours): Speaking with Intent
            dialog_user = [{"role": "user", "content": f"""
Speak with intent and summarize the following article.\n
{article}
            """.strip()}]
        else:  # Baseline: LLM Generation without Intent
            dialog_user = [{"role": "user", "content": f"""
Summarize the following article.\n
{article}
            """.strip()}]

        # Answer Trigger Prompting methods
        if use_cot:  # Zero-shot Chain-of-Thought (CoT) prompting  https://arxiv.org/abs/2205.11916
            dialog_user[0]["content"] = dialog_user[0]["content"] + "\n\n" + f"""
Let's think step by step.
            """.strip()
        elif use_ps:  # Plan-and-Solve (PS) prompting  https://aclanthology.org/2023.acl-long.147/
            # RAW: Let's first understand the problem and devise a plan to solve the problem. \
            # Then, let's carry out the plan and solve the problem step by step.
            dialog_user[0]["content"] = dialog_user[0]["content"] + "\n\n" + f"""
Let's first understand the article and devise a plan to solve the task. \
Then, let's carry out the plan and summarize the article step by step.
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
                "task_type": "sum",
                "article": article,
            }
        }
        return result_dict
