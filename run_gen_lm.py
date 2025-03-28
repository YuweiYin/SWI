#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import time
import json
from typing import Optional

import fire
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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


class LMGen:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: dict,
            seed: int = 42,
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
            hf_id: str = "meta-llama/Llama-3.1-8B-Instruct",
            bsz: int = 1,
            show_generation: bool = False,
            debug: bool = False,
            output_dir: Optional[str] = None,
            max_gen_len: int = 4096,
            gen_temperature: float = 0.0,
            use_swi: bool = False,
            use_cot: bool = False,
            use_arr: bool = False,
            use_ps: bool = False,
    ):
        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.seed = seed
        self.hf_id = hf_id
        self.hf_name = "--".join(hf_id.split("/"))
        self.show_generation = show_generation  # If True, show outputs during generation
        self.debug = debug

        if isinstance(project_dir, str) and os.path.isdir(project_dir):
            self.project_dir = project_dir
        else:
            self.project_dir = os.getcwd()
        assert os.path.isdir(project_dir)

        self.output_dir = output_dir
        self.bsz = bsz
        self.max_gen_len = max_gen_len
        self.gen_temperature = gen_temperature
        self.use_swi = use_swi
        self.use_cot = use_cot
        self.use_arr = use_arr
        self.use_ps = use_ps

        self.task_class_dict = {
            # Mathematical Reasoning (Math)
            "gsm8k": EvalTaskGSM8K,
            "gsm8k_platinum": EvalTaskGSM8KPlatinum,
            "math500": EvalTaskMATH500,
            "amc23": EvalTaskAMC23,
            "aime24": EvalTaskAIME24,
            "aime25": EvalTaskAIME25,
            # Multiple-choice Question Answering (MCQA)
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

        # Tokenizer and LLM model
        self.tokenizer = self.load_tokenizer(model_path=self.model_path, padding_side="left", truncation_side="left")
        self.terminators_gen = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        ]
        self.terminators_gen_set = set(self.terminators_gen)
        self.terminators_gen = list(self.terminators_gen_set)
        self.model = None

    @staticmethod
    def _handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    def load_tokenizer(
            self,
            model_path,
            padding_side="left",
            truncation_side="left",
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side=padding_side,
            truncation_side=truncation_side,  # "right" for training, "left" for generating
            cache_dir=self.cache_dir,
            # local_files_only=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        max_len = tokenizer.max_len_single_sentence
        if self.verbose:
            self.logger.info(
                f">>> len(tokenizer.vocab) = {len(tokenizer.vocab)}; "
                f"tokenizer.max_len_single_sentence = {max_len}")  # LLaMA-3: 131071

        return tokenizer

    def load_model(
            self,
            model_path,
            tokenizer,
    ):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # torch.bfloat16
            device_map="auto",  # !pip install accelerate
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            # local_files_only=True,
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id  # eos_token_id
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.verbose:
            self.logger.info(f">>> Base Model loaded: {model_path}")
            self.logger.info(f">>> [Base Model] Number of total parameters: {total_params}")
            self.logger.info(f">>> [Base Model] Number of trainable parameters: {train_params}")

        return model

    def run_generation(
            self,
            prompts,
            model,
            tokenizer,
            need_tokenize: bool = True,
            new_gen_unit: int = 512,
            gen_until_eot: bool = True,
    ) -> dict:
        if need_tokenize:
            input_ids = self.tokenizer(
                prompts,
                padding=True,  # truncation=True, max_length=1024
                return_tensors="pt",
            ).to(model.device)  # batch_size=1
        else:
            input_ids = prompts
            input_ids = input_ids.to(model.device)
        len_input = input_ids.data["input_ids"].size(-1)

        # As the batch size is 1, we do not need the attention masks
        cur_input_ids = input_ids["input_ids"]

        end_with_eot = False
        self.max_gen_len = max(self.max_gen_len, new_gen_unit)
        new_gen_cnt = 0
        with torch.no_grad():
            while not end_with_eot:
                # https://huggingface.co/docs/transformers/en/main_classes/text_generation
                outputs = model.generate(
                    # **input_ids,
                    cur_input_ids,
                    max_new_tokens=new_gen_unit,
                    eos_token_id=self.terminators_gen,
                    do_sample=self.gen_temperature > 0.0,
                    # do_sample=True,  # False: greedy decoding (the most deterministic)
                    temperature=self.gen_temperature if self.gen_temperature > 0.0 else None,  # defaults to 1.0
                    # top_p=0.9,  # defaults to 1.0
                    # output_attentions=False,
                    # output_hidden_states=False,
                    # output_scores=True,
                    output_logits=True,
                    return_dict_in_generate=True,
                )  # Produce at most `new_gen_unit` tokens for each generation run

                # Check the last token of the current output
                output_ids = outputs["sequences"]
                last_token_id = int(output_ids[0][-1].cpu().item())
                if last_token_id in self.terminators_gen_set:
                    end_with_eot = True
                    break

                new_gen_cnt += new_gen_unit  # record the number of generated tokens
                if new_gen_cnt >= self.max_gen_len:  # No more than `self.max_gen_len` (to avoid GPU OOM)
                    break
                if not gen_until_eot:  # `gen_until_eot` means we will generate tokens until end-of-text
                    break

                cur_input_ids = output_ids  # Next round, generate new tokens after the current output

        output_ids = outputs["sequences"]
        output_text = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        input_text = tokenizer.batch_decode(
            input_ids["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        assert len(input_text) == len(prompts) == len(output_text)
        output_text_pure = []
        for _input, _prompt, _output in zip(input_text, prompts, output_text):
            output_pure = _output[len(_input):]
            output_text_pure.append(output_pure)
            if self.verbose and self.show_generation:
                self.logger.info("================================== >>> output <<<")
                self.logger.info(output_pure)

        return {
            "prompts": prompts,
            "len_input": len_input,
            "input_text": input_text,
            "outputs": outputs,
            "output_text": output_text_pure,
            "end_with_eot": end_with_eot,
        }

    def run_language_modeling(
            self,
            prompts,
            model,
            tokenizer,
            need_tokenize: bool = True,
    ) -> dict:
        if need_tokenize:
            input_ids = self.tokenizer(
                prompts,
                padding=True,  # truncation=True, max_length=1024
                return_tensors="pt",
            ).to(model.device)  # batch_size=1
        else:
            input_ids = prompts
            input_ids = input_ids.to(model.device)
        len_input = input_ids.data["input_ids"].size(-1)
        target_ids = input_ids["input_ids"].to(model.device)
        input_ids.data["labels"] = target_ids

        with torch.no_grad():
            outputs = model(**input_ids)
        output_ids = outputs["logits"].argmax(-1)
        output_text = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        input_text = tokenizer.batch_decode(
            input_ids["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        assert len(input_text) == len(prompts) == len(output_text)
        output_text_pure = []
        for _input, _prompt, _output in zip(input_text, prompts, output_text):
            output_pure = _output[len(_input):]
            output_text_pure.append(output_pure)
            if self.verbose and self.show_generation:
                self.logger.info("================================== >>> output <<<")
                self.logger.info(output_pure)

        return {
            "prompts": prompts,
            "len_input": len_input,
            "input_text": input_text,
            "outputs": outputs,
            "output_text": output_text_pure,
        }

    def lm_generate(
            self,
            eval_task_name: str,
    ):
        # Generation Phase: load datasets, load the model, set prompts, freely generation,
        #   and save results to JSON files (task/dataset information, input, and output)
        #   [apply chat templates for Chat models]

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        assert isinstance(self.output_dir, str), "Please specify --output_dir"
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

        # Load the model
        if self.model is None:
            model = self.load_model(model_path=self.model_path, tokenizer=self.tokenizer)
            self.model = model
        else:
            model = self.model

        # Deal with each task (and sub-tasks)
        all_results = {}
        show_cnt = 100
        for dataset_dict in dataset_list:
            cur_results = []
            ds_name, subset = dataset_dict["hf_dataset"], dataset_dict["hf_subset"]
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

            if "options" in dataset_dict:
                ds_options = list(dataset_dict["options"])
            else:
                ds_options = []

            # Run generation with batch_size = 1
            for idx, data_item in enumerate(eval_dataset):
                assert isinstance(data_item, dict)
                data_item["__ds_options"] = ds_options

                prompt_dict = eval_task_obj.set_dialog(
                    ds_name=ds_name,
                    subset=subset,
                    data_item=data_item,
                    use_swi=self.use_swi,
                    use_cot=self.use_cot,
                    use_arr=self.use_arr,
                    use_ps=self.use_ps,
                )
                item_info = prompt_dict["info"]

                # Run generation (batch_size = 1)
                cur_prompt = self.tokenizer.apply_chat_template(
                    prompt_dict["dialog"],
                    tokenize=False,
                    padding=False,
                    add_generation_prompt=True,
                    return_tensors=None
                )
                assert isinstance(cur_prompt, str)
                gen_dict = self.run_generation(
                    prompts=[cur_prompt], model=model, tokenizer=self.tokenizer, need_tokenize=True)
                cur_gen_output = {
                    "index": idx,
                    "prompt": cur_prompt,  # The input prompt
                    "len_input": int(gen_dict["len_input"]),  # Number of tokens of the model input/prompt
                    "output_text": str(gen_dict["output_text"][0]).strip(),  # The LLM output (excluding the input)
                    "answers": prompt_dict["answers"],  # The golden answers: List[str]
                    "end_with_eot": bool(gen_dict["end_with_eot"]),  # True if the output ends with end-of-text
                    "info": item_info,
                }

                cur_results.append(cur_gen_output)
                if self.verbose and len(cur_results) % show_cnt == 0:
                    self.logger.info(f">>> Progress: [{ds_id}] [{len(cur_results)} / {len_dataset}]")

            all_results[ds_id] = cur_results

        # Save the generation outputs and show logs
        output_dir = os.path.join(self.output_dir, eval_task_name, self.hf_name)
        os.makedirs(output_dir, exist_ok=True)
        output_fp = os.path.join(output_dir, "results_gen.json")
        if os.path.exists(output_fp):
            self.logger.info(f"Results will be overwritten: {output_fp}")
        else:
            self.logger.info(f"Results will be saved at: {output_fp}")
        dumped = json.dumps(
            all_results,
            indent=2,  # indent=None,
            default=self._handle_non_serializable,
            ensure_ascii=True,
        )
        with open(output_fp, "w", encoding="utf-8") as fp_out:
            fp_out.write(dumped)
        self.logger.info(
            f">>> DONE ALL. hf_id = {self.hf_id}; model_path = {self.model_path}\n"
            f"use_swi: {self.use_swi}, gen_temperature: {self.gen_temperature}, batch_size: {self.bsz}"
        )


def main(
    task: int = 0,
    eval_task_name: Optional[str] = None,
    hf_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    seed: int = 42,
    cuda: Optional[str] = None,
    bsz: int = 1,
    verbose: bool = False,
    debug: bool = False,
    output_dir: Optional[str] = None,
    max_gen_len: int = 4096,
    gen_temperature: float = 0.0,
    use_swi: bool = False,
    use_cot: bool = False,
    use_arr: bool = False,
    use_ps: bool = False,
    **kwargs
) -> None:
    """
    [Stage 1: Reasoning & Answer Generation]
    Run LLM Generation on different tasks and benchmarks. Store the answers.
    (zero-shot generation w/ or w/o SWI: Speaking with Intent)

    :param task: 1. language model generation.
    :param eval_task_name: The name(s) of the evaluation task. (e.g., "xsum", "xlsum", and "gsm8k,math500")
    :param hf_id: ORGANIZATION_NAME/MODEL_NAME, e.g., "meta-llama/Llama-3.1-8B-Instruct"
    :param cache_dir: The root directory of the cache.
    :param project_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param bsz: The batch size.
    :param verbose: Verbose mode: show logs.
    :param debug: Debugging / developing mode.
    :param output_dir: The path to the output file where the result metrics will be saved.
    :param max_gen_len: The maximum number of newly generated tokens.
    :param gen_temperature: The temperature used in LLM generation. Default: 0.0
    :param use_swi: Use our SWI method or not. SWI: Speaking with Intent
    :param use_cot: Use zero-shot Chain-of-Thought prompting method or not.  https://arxiv.org/abs/2205.11916
    :param use_arr: Use ARR method or not. ARR: Analyzing, Retrieving, and Reasoning.  https://arxiv.org/abs/2502.04689
    :param use_ps: Use Plan-and-Solve method or not.  https://aclanthology.org/2023.acl-long.147/
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("LM_Gen")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Unused parameters in kwargs: {kwargs}\n")

    if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
        os.environ["HF_HOME"] = cache_dir
        # os.environ["HF_HOME"] = os.path.join(cache_dir, "datasets")
        # os.environ["HF_HOME"] = os.path.join(cache_dir, "hub")
    else:
        cache_dir = None

    lm_gen = LMGen(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        cache_dir=cache_dir,
        project_dir=project_dir,
        hf_id=hf_id,
        bsz=max(int(bsz), 1),
        debug=debug,
        output_dir=output_dir,
        max_gen_len=int(max_gen_len),
        gen_temperature=float(gen_temperature),
        use_swi=use_swi,
        use_cot=use_cot,
        use_arr=use_arr,
        use_ps=use_ps,
    )

    task = int(task)
    match task:
        case 1:
            if isinstance(eval_task_name, tuple) or isinstance(eval_task_name, list):
                for cur_task_name in eval_task_name:
                    cur_task_name = str(cur_task_name).strip()
                    logger.info(f">>> <START> {cur_task_name}\n")
                    lm_gen.lm_generate(eval_task_name=cur_task_name)
                    logger.info(f">>> <END> {cur_task_name}\n\n\n")
            elif isinstance(eval_task_name, str):
                eval_task_name = str(eval_task_name).strip()
                logger.info(f">>> <START> {eval_task_name}\n")
                lm_gen.lm_generate(eval_task_name=eval_task_name)
                logger.info(f">>> <END> {eval_task_name}\n\n\n")
            else:
                raise ValueError(f"--eval_task_name should be a tuple/list/str: {eval_task_name}")
        case _:
            raise ValueError(f"ValueError: task = {task}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
