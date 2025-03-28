# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

from typing import Optional, Dict, Any

from tasks import EvalTaskManager


class EvalTaskSamSum(EvalTaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
    ):
        super().__init__(verbose, logger, cache_dir, project_dir)

        # SAMSum: Summarization
        # Train = 14732, Valid = 818, Test = 819
        # Features: ["id", "dialogue", "summary"]
        # Eval: test set
        # >>> [use_swi = False] >>> #Sub-Tasks = 1; #Total Ins. = 819; avg_len_token: 209.427; std_len_token: 104.665
        # >>> [use_swi = True] >>> #Sub-Tasks = 1; #Total Ins. = 819; avg_len_token: 292.427; std_len_token: 104.665

        self.task_name = "samsum"
        self.task_info = {
            "hf_dataset": [  # [hf_id, subset, eval_set]
                ["Samsung/samsum", None, "test"],
            ],
        }

        self.system_prompt_raw = f"""
You are a helpful assistant. \
You are good at summarizing documents and the summary must start with "Final Summary:"
        """.strip()
        self.system_prompt_swi = f"""
You are a helpful assistant who speaks with intent. \
You are good at summarizing documents and the summary must start with "Final Summary:"
During generation, follow all the requirements below:
1. Always explicitly state your own intent before speaking each sentence.
2. Each intent statement should explain the sentence followed up.
3. Your intent must start with the "<INTENT>" tag and end with the "</INTENT>" tag.
4. At last, clearly and concisely give your final summary starting with "Final Summary:"
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

        # Load data
        hf_ds_list = self.task_info["hf_dataset"]
        assert isinstance(hf_ds_list, list) and len(hf_ds_list) > 0

        dialog_sys = []
        if use_swi:
            dialog_sys.append({"role": "system", "content": self.system_prompt_swi})
        else:
            dialog_sys.append({"role": "system", "content": self.system_prompt_raw})

        # Process data ["id", "dialogue", "summary"]
        article = str(data_item["dialogue"]).strip().replace("\n\n", "\n")
        summary = str(data_item["summary"]).strip()
        answers = [summary]

        # Set the main prompt (zero-shot)
        if use_swi:  # SWI (ours): Speaking with Intent
            dialog_user = [{"role": "user", "content": f"""
Speak with intent and summarize the following dialogue.\n
{article}
            """.strip()}]
        else:  # Baseline: LLM Generation without Intent
            dialog_user = [{"role": "user", "content": f"""
Summarize the following dialogue.\n
{article}
            """.strip()}]

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
