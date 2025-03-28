#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import re
import sys
import time
import json
import string
from typing import Optional, List

import fire
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

import nltk
import evaluate
from datasets.download.download_config import DownloadConfig
from evaluate_metrics.sentence_transformers import SentenceTransformer

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


class LMEval:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: dict,
            seed: int = 42,
            eval_task_name="",
            eval_metric_name="ALL",
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
            hf_id: str = "meta-llama/Llama-3.1-8B-Instruct",
            bsz: int = 1,
            show_generation: bool = False,
            debug: bool = False,
            output_dir: Optional[str] = None,
            overwrite: bool = False,
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

        self.eval_task_name = eval_task_name
        self.eval_metric_name = eval_metric_name
        self.output_dir = output_dir
        self.bsz = bsz
        self.overwrite = overwrite

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
        self.model = None

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

        # Evaluators
        hf_eval_cache = os.path.join(self.cache_dir, "evaluate")
        os.makedirs(hf_eval_cache, exist_ok=True)
        download_config = DownloadConfig(cache_dir=hf_eval_cache, force_download=False)
        try:
            self.eval_bleu = evaluate.load(
                path="evaluate_metrics/bleu", cache_dir=hf_eval_cache, download_config=download_config)
            self.eval_sacrebleu = evaluate.load(
                path="evaluate_metrics/sacrebleu", cache_dir=hf_eval_cache, download_config=download_config)
            self.eval_rouge = evaluate.load(
                path="evaluate_metrics/rouge", cache_dir=hf_eval_cache, download_config=download_config)
            self.eval_meteor = evaluate.load(
                path="evaluate_metrics/meteor", cache_dir=hf_eval_cache, download_config=download_config)
            self.eval_chrf = evaluate.load(
                path="evaluate_metrics/chrf", cache_dir=hf_eval_cache, download_config=download_config)
            self.eval_bertscore = evaluate.load(
                path="evaluate_metrics/bertscore", cache_dir=hf_eval_cache, download_config=download_config)
        except Exception as e:
            if self.verbose:
                self.logger.info(e)  # Potentially, urllib3.connection.HTTPSConnection - ConnectionError
            self.eval_bleu = None
            self.eval_sacrebleu = None
            self.eval_rouge = None
            self.eval_meteor = None
            self.eval_chrf = None
            self.eval_bertscore = None
            if self.eval_task_name in self.sum_set:
                sys.exit(1)
        self.sbert_model_en = None  # English-only Sentence Transformer
        self.sbert_model_x = None   # Multilingual Sentence Transformer

        # ["em", "rouge", "sacrebleu", "meteor", "chrf", "bertscore", "sentence_bert"]
        # Note: BLEU, SacreBLEU, METEOR, and chrF are mainly used for machine translation.
        #   BERTScore always returns scores near 85%, and Sentence BERT always outputs 99%.
        #   Hence, we do not use BERTScore and Sentence BERT. Instead, we try LLM-as-a-Judge for semantic matching.
        self.task_eval_dict = {
            # Mathematical Reasoning. Key metrics: "em"
            "gsm8k": ("math", ["em"]),
            "gsm8k_platinum": ("math", ["em"]),
            "math500": ("math", ["em"]),
            "amc23": ("math", ["em"]),
            "aime24": ("math", ["em"]),
            "aime25": ("math", ["em"]),
            # Question Answering. Key metrics: "em" and "mcqa"
            "logiqa": ("qa", ["em", "mcqa"]),
            "commonsense_qa": ("qa", ["em", "mcqa"]),
            "social_iqa": ("qa", ["em", "mcqa"]),
            "openbookqa": ("qa", ["em", "mcqa"]),
            "ai2_arc": ("qa", ["em", "mcqa"]),
            "bbh": ("qa", ["em", "mcqa"]),
            "mmlu": ("qa", ["em", "mcqa"]),
            "mmlu_pro": ("qa", ["em", "mcqa"]),
            # Summarization. Key metrics: "rouge"
            "cnn_dailymail": ("sum", ["rouge"]),
            "xsum": ("sum", ["rouge"]),
            "xlsum": ("sum", ["rouge"]),
            "samsum": ("sum", ["rouge"]),
            "dialogsum": ("sum", ["rouge"]),
            "wiki_lingua": ("sum", ["rouge"]),
        }

        self.metric_func = {
            "em": self.compute_exact_match,
            "bleu": self.compute_bleu,
            "sacrebleu": self.compute_sacrebleu,
            "rouge": self.compute_rouge,
            "meteor": self.compute_meteor,
            "chrf": self.compute_chrf,
            "bertscore": self.compute_bert_score,
            "sentence_bert": self.compute_sentence_bert,
            "mcqa": self.compute_multi_choice_qa,
        }

        self.code2lang = {
            "en": "English", "de": "German", "fr": "French", "zh": "Chinese",
            "et": "Estonian", "hi": "Hindi", "tr": "Turkish",
        }
        self.lang2code = {v: k for k, v in self.code2lang.items()}
        self.punc_remover = str.maketrans("", "", string.punctuation)  # r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        self.space_remover = str.maketrans("", "", string.whitespace)  # " \t\n\r\v\f"

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
        # tokenizer.add_special_tokens({"pad_token": "<|pad_of_text|>"})
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
                padding=True,  # truncation=True, max_length=1024,
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

    def compute_exact_match(
            self,
            # prediction: str,
            references: List[str],
            # item_info: dict = None,
            # eval_task_name: str,
            **kwargs
    ) -> dict:
        assert "pred_final" in kwargs
        pred_final = str(kwargs["pred_final"]).strip()
        references = [str(_ref).strip() for _ref in references]

        # Matching anyone in the references will have an EM score of 1; otherwise 0.
        for ref in references:
            if pred_final == ref:
                return {"score": float(1.0)}

        # Normalize strings and then match
        pred_final_new, references_new = pred_final, references
        pred_final_new = pred_final_new.translate(self.punc_remover).strip()  # Remove all punctuations
        pred_final_new = pred_final_new.translate(self.space_remover).strip()  # Remove all whitespaces
        references_new = [_ref.translate(self.punc_remover).strip() for _ref in references_new]
        references_new = [_ref.translate(self.space_remover).strip() for _ref in references_new]
        for ref in references_new:
            if pred_final_new == ref:
                return {"score": float(1.0)}

        return {"score": float(0.0)}

    def compute_bleu(
            self,
            # prediction: str,
            references: List[str],
            # item_info: dict = None,
            # eval_task_name: str,
            **kwargs
    ) -> dict:
        """
        - https://aclanthology.org/P02-1040/
        - https://huggingface.co/spaces/evaluate-metric/bleu

        - predictions (list of strs): Translations to score.
        - references (list of lists of strs): references for each translation.
        - ** tokenizer** : approach used for standardizing predictions and references. The default tokenizer is `tokenizer_13a`
        - max_order (int): Maximum n-gram order to use when computing BLEU score. Defaults to 4.
        - smooth (boolean): Whether or not to apply Lin et al. 2004 smoothing. Defaults to False.
        Lin et al. 2004. https://aclanthology.org/C04-1072/
        """

        assert "pred_final" in kwargs
        pred_final = str(kwargs["pred_final"]).strip()
        references = [str(_ref).strip() for _ref in references]

        res_dict = self.eval_bleu.compute(predictions=[pred_final], references=[references], max_order=4, smooth=False)
        res_dict["score"] = float(res_dict["bleu"])
        return res_dict

    def compute_sacrebleu(
            self,
            # prediction: str,
            references: List[str],
            # item_info: dict = None,
            # eval_task_name: str,
            **kwargs
    ) -> dict:
        """
        - https://aclanthology.org/W18-6319/
        - https://huggingface.co/spaces/evaluate-metric/sacrebleu

        - predictions (List[str]): list of translations to score. Each translation should be tokenized into a list of tokens.
        - references (List[List[str]]): A list of lists of references. The contents of the first sub-list are the references for the first prediction, the contents of the second sub-list are for the second prediction, etc. Note that there must be the same number of references for each prediction (i.e. all sub-lists must be of the same length).
        - smooth_method (str): The smoothing method to use, defaults to "exp". Possible values are:
            - "none": no smoothing
            - "floor": increment zero counts
            - "add-k": increment num/denom by k for n>1
            - "exp": exponential decay
        - smooth_value (float): The smoothing value. Only valid when smooth_method="floor" (in which case smooth_value defaults to 0.1) or smooth_method="add-k" (in which case smooth_value defaults to 1).
        - tokenize (str): Tokenization method to use for BLEU. If not provided, defaults to "zh" for Chinese, "ja-mecab" for Japanese and "13a" (mteval) otherwise. Possible values are:
            - "none": No tokenization.
            - "zh": Chinese tokenization.
            - "13a": mimics the mteval-v13a script from Moses.
        - "intl": International tokenization, mimics the mteval-v14 script from Moses
        - "char": Language-agnostic character-level tokenization.
        - "ja-mecab": Japanese tokenization. Uses the MeCab tokenizer.
        - lowercase (bool): If True, lowercases the input, enabling case-insensitivity. Defaults to False.
        - force (bool): If True, insists that your tokenized input is actually detokenized. Defaults to False.
        - use_effective_order (bool): If True, stops including n-gram orders for which precision is 0. This should be True, if sentence-level BLEU will be computed. Defaults to False.
        """

        assert "pred_final" in kwargs
        pred_final = str(kwargs["pred_final"]).strip()
        references = [str(_ref).strip() for _ref in references]

        res_dict = self.eval_sacrebleu.compute(predictions=[pred_final], references=[references])
        res_dict["sacrebleu"] = res_dict["score"]
        res_dict["score"] = float(res_dict["sacrebleu"]) / 100.0  # from [0, 100] to [0, 1]
        return res_dict

    def compute_rouge(
            self,
            # prediction: str,
            references: List[str],
            # item_info: dict = None,
            # eval_task_name: str,
            **kwargs
    ) -> dict:
        """
        - https://aclanthology.org/W04-1013/
        - https://huggingface.co/spaces/evaluate-metric/rouge

        - predictions (list): list of predictions to score. Each prediction should be a string with tokens separated by spaces.
        - references (list or List[list]): list of reference for each prediction or a list of several references per prediction. Each reference should be a string with tokens separated by spaces.
        - rouge_types (list): A list of rouge types to calculate. Defaults to ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'].
            - "rouge1": unigram (1-gram) based scoring
            - "rouge2": bigram (2-gram) based scoring
            - "rougeL": Longest common subsequence based scoring.
            - "rougeLSum": splits text using line breaks
            - See [here](https://github.com/huggingface/datasets/issues/617) for more information
        - use_aggregator (boolean): If True, returns aggregates. Defaults to True.
        - use_stemmer (boolean): If True, uses Porter stemmer to strip word suffixes. Defaults to False.
        """

        assert "pred_final" in kwargs
        pred_final = str(kwargs["pred_final"]).strip()
        references = [str(_ref).strip() for _ref in references]

        res_dict = self.eval_rouge.compute(predictions=[pred_final], references=[references])
        res_dict["score"] = float(np.mean(list(res_dict.values())).item())
        return res_dict

    def compute_meteor(
            self,
            # prediction: str,
            references: List[str],
            # item_info: dict = None,
            # eval_task_name: str,
            **kwargs
    ) -> dict:
        """
        - https://aclanthology.org/W05-0909/
        - https://huggingface.co/spaces/evaluate-metric/meteor

        - METEOR has two mandatory arguments:
            - predictions: a list of predictions to score. Each prediction should be a string with tokens separated by spaces.
            - references: a list of references (in the case of one reference per prediction)
        - It also has several optional parameters:
            - alpha: Parameter for controlling relative weights of precision and recall. The default value is 0.9.
            - beta: Parameter for controlling shape of penalty as a function of fragmentation. The default value is 3.
            - gamma: The relative weight assigned to fragmentation penalty. The default is 0.5.
        """

        assert "pred_final" in kwargs
        pred_final = str(kwargs["pred_final"]).strip()
        references = [str(_ref).strip() for _ref in references]

        res_dict = self.eval_meteor.compute(predictions=[pred_final], references=[references])
        res_dict["score"] = float(res_dict["meteor"])
        return res_dict

    def compute_chrf(
            self,
            # prediction: str,
            references: List[str],
            # item_info: dict = None,
            # eval_task_name: str,
            **kwargs
    ) -> dict:
        """
        - https://aclanthology.org/W15-3049/
        - https://huggingface.co/spaces/evaluate-metric/chrf

        - predictions (List[str]): The predicted sentences.
        - references (List[List[str]]): The references. There should be one reference sub-list for each prediction sentence.
        - char_order (int): Character n-gram order. Defaults to 6.
        - word_order (int): Word n-gram order. If equals to 2, the metric is referred to as chrF++. Defaults to 0.
        - beta (int): Determine the importance of recall w.r.t precision. Defaults to 2.
        - lowercase (bool): If True, enables case-insensitivity. Defaults to False.
        - whitespace (bool): If True, include whitespaces when extracting character n-grams. Defaults to False.
        - eps_smoothing (bool): If True, applies epsilon smoothing similar to reference chrF++.py, NLTK, and Moses implementations. If False, takes into account effective match order similar to sacreBLEU < 2.0.0. Defaults to False.
        """

        assert "pred_final" in kwargs
        pred_final = str(kwargs["pred_final"]).strip()
        references = [str(_ref).strip() for _ref in references]

        # chrF++: word_order=2
        # res_dict = self.eval_chrf.compute(predictions=[pred_final], references=[references], word_order=0)
        res_dict = self.eval_chrf.compute(predictions=[pred_final], references=[references], word_order=2)
        res_dict["chrf"] = res_dict["score"]
        res_dict["score"] = float(res_dict["score"]) / 100.0
        return res_dict

    def compute_bert_score(
            self,
            # prediction: str,
            references: List[str],
            # item_info: dict = None,
            # eval_task_name: str,
            **kwargs
    ) -> dict:
        """
        - https://openreview.net/forum?id=SkeHuCVFDr
        - https://huggingface.co/spaces/evaluate-metric/bertscore

        - num_layers (int): The layer of representation to use. The default is the number of layers tuned on WMT16 correlation data, which depends on the model_type used.
        - verbose (bool): Turn on intermediate status update. The default value is False.
        - idf (bool or dict): Use idf weighting; can also be a precomputed idf_dict.
        - device (str): On which the contextual embedding model will be allocated on. If this argument is None, the model lives on cuda:0 if cuda is available.
        - nthreads (int): Number of threads used for computation. The default value is 4.
        - rescale_with_baseline (bool): Rescale BERTScore with the precomputed baseline. The default value is False.
        - batch_size (int): BERTScore processing batch size, at least one of model_type or lang. lang needs to be specified when rescale_with_baseline is True.
        - baseline_path (str): Customized baseline file.
        - use_fast_tokenizer (bool): use_fast parameter passed to HF tokenizer. The default value is False.
        """

        assert "pred_final" in kwargs
        pred_final = str(kwargs["pred_final"]).strip()
        references = [str(_ref).strip() for _ref in references]

        assert "lang_code" in kwargs
        lang_code = str(kwargs["lang_code"]).strip()

        # Language Code: https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
        # Model Options: "roberta-large" (default), "xlm-roberta-large", "distilbert-base-uncased"
        if lang_code == "en":  # English-only RoBERTa  https://huggingface.co/FacebookAI/roberta-large
            res_dict = self.eval_bertscore.compute(
                predictions=[pred_final], references=[references], lang=lang_code, model_type="roberta-large",
                batch_size=self.bsz, device=self.cuda_dict["device"], cache_dir=self.cache_dir
            )
        else:  # Multilingual XLM-RoBERTa  https://huggingface.co/FacebookAI/xlm-roberta-large
            res_dict = self.eval_bertscore.compute(
                predictions=[pred_final], references=[references], lang=lang_code, model_type="xlm-roberta-large",
                batch_size=self.bsz, device=self.cuda_dict["device"], cache_dir=self.cache_dir
            )

        assert isinstance(res_dict["f1"], list) and len(res_dict["f1"]) == 1
        res_dict["score"] = float(res_dict["f1"][0])
        return res_dict

    def compute_sentence_bert(
            self,
            # prediction: str,
            references: List[str],
            # item_info: dict = None,
            # eval_task_name: str,
            **kwargs
    ) -> dict:
        """
        - https://aclanthology.org/D19-1410/
        - https://aclanthology.org/2020.emnlp-main.365/
        - https://www.sbert.net/
        - https://github.com/UKPLab/sentence-transformers
        """

        assert "pred_final" in kwargs
        pred_final = str(kwargs["pred_final"]).strip()
        references = [str(_ref).strip() for _ref in references]

        assert "lang_code" in kwargs
        lang_code = str(kwargs["lang_code"]).strip()

        # Load the pretrained Sentence Transformer model
        if lang_code == "en":  # English-only
            if self.sbert_model_en is None:
                # model = SentenceTransformer("roberta-large")
                model = SentenceTransformer(
                    model_name_or_path="roberta-large", cache_folder=self.cache_dir,
                    trust_remote_code=True, device=self.cuda_dict["device"])
                self.sbert_model_en = model
            else:
                model = self.sbert_model_en
        else:  # Multilingual
            if self.sbert_model_x is None:
                # model = SentenceTransformer("xlm-roberta-large")
                model = SentenceTransformer(
                    model_name_or_path="xlm-roberta-large", cache_folder=self.cache_dir,
                    trust_remote_code=True, device=self.cuda_dict["device"])
                self.sbert_model_x = model
            else:
                model = self.sbert_model_x

        # Computer the similarity of each pred-ref pairs
        pred_embedding = model.encode([pred_final])
        ref_embeddings = model.encode(references)
        similarities = model.similarity(pred_embedding, ref_embeddings)

        # Return the highest similarity score
        score = float(similarities.cpu().numpy().max().item())
        return {"score": score}

    def compute_multi_choice_qa(
            self,
            prediction: str,
            # references: List[str],
            item_info: dict = None,
            # eval_task_name: str,
            **kwargs
    ) -> dict:
        """
        - https://arxiv.org/abs/2502.04689
        - https://github.com/YuweiYin/ARR
        """

        # assert "pred_final" in kwargs
        # pred_final = str(kwargs["pred_final"]).strip()
        # references = [str(_ref).strip() for _ref in references]

        # Load the options and the index of the correct answer
        assert isinstance(item_info, dict) and "options" in item_info and "ans_idx" in item_info
        options, ans_idx = item_info["options"], item_info["ans_idx"]
        # assert isinstance(options, list) and len(options) > 0 and 0 <= ans_idx < len(options)
        if not (isinstance(options, list) and len(options) > 0 and
                isinstance(ans_idx, int) and 0 <= ans_idx < len(options)):
            return {"score": float(0.0)}

        options = [str(_op).strip() for _op in options]

        # Accuracy score: select the option with the lowest LLM perplexity / avg nll_loss
        assert "gen_prompt" in kwargs
        gen_prompt = str(kwargs["gen_prompt"]).strip()
        concat_prompts = [f"{gen_prompt}\n{prediction}\nFinal Answer: {_op}" for _op in options]

        # Load the model (only the first time)
        if self.model is None:
            model = self.load_model(model_path=self.model_path, tokenizer=self.tokenizer)
            self.model = model
        else:
            model = self.model

        # Run language modeling (batch_size = 1) --> obtain logits / nll loss / perplexity
        eval_losses = []
        for concat_prompt in concat_prompts:
            eval_dict = self.run_language_modeling(
                prompts=[concat_prompt], model=model, tokenizer=self.tokenizer, need_tokenize=True)
            eval_losses.append(eval_dict["outputs"]["loss"].cpu().detach().numpy().item())

        eval_choice = int(np.argmin(eval_losses).item())
        if eval_choice == ans_idx:
            score = 1.0
        else:
            score = 0.0

        return {"score": float(score)}

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

    def lm_evaluate(
            self,
            eval_task_name: str,
    ):
        # Evaluation Phase: load the result JSON file, extract reasoning/intent and answers, and compute scores

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load the generation outputs
        assert isinstance(self.output_dir, str) and os.path.isdir(self.output_dir), "Please specify --output_dir"
        output_dir = os.path.join(self.output_dir, eval_task_name, self.hf_name)
        output_fp = os.path.join(output_dir, "results_gen.json")
        if not os.path.isfile(output_fp):
            self.logger.info(
                f">>> hf_id = {self.hf_id}; model_path = {self.model_path}\n"
                f"output_dir: {output_dir}\n"
                f">>> !!! >>> [SKIP; No --output_fp] output_fp does not exist: {output_fp}"
            )
            return
        with open(output_fp, "r", encoding="utf-8") as fp_in:
            gen_results = json.load(fp_in)

        # Set the saving filepath
        if isinstance(self.eval_metric_name, str):
            output_eval_fp = os.path.join(output_dir, f"results_eval-{eval_task_name}-{self.eval_metric_name}.json")
        else:
            output_eval_fp = os.path.join(output_dir, f"results_eval-{eval_task_name}.json")
        if os.path.exists(output_eval_fp):
            if self.overwrite:
                self.logger.info(f"Results will be overwritten: {output_eval_fp}")
            else:
                self.logger.info(
                    f">>> hf_id = {self.hf_id}; model_path = {self.model_path}\n"
                    f"output_dir: {output_dir}\n"
                    f">>> !!! >>> [SKIP; No --overwrite] File already exists: {output_eval_fp}"
                )
                return
        else:
            self.logger.info(f"Results will be saved at: {output_eval_fp}")

        assert eval_task_name in self.task_class_dict, \
            f"AssertionError: task name {eval_task_name} not in task_class_dict"
        eval_task_class = self.task_class_dict[eval_task_name]

        eval_task_obj = eval_task_class(
            verbose=self.verbose,
            logger=self.logger,
            cache_dir=self.cache_dir,
            project_dir=self.project_dir,
        )

        assert eval_task_name in self.task_eval_dict, \
            f"AssertionError: task name {eval_task_name} not in task_eval_dict"
        if self.eval_metric_name is None or self.eval_metric_name == "ALL":
            eval_metrics = list(self.task_eval_dict[eval_task_name][1])  # Use all default metrics for the task
        elif isinstance(self.eval_metric_name, str) and self.eval_metric_name in self.metric_func:
            eval_metrics = [self.eval_metric_name]
        elif isinstance(self.eval_metric_name, tuple) or isinstance(self.eval_metric_name, list):
            eval_metrics = [e_m for e_m in self.eval_metric_name if e_m in self.metric_func]
        else:
            raise ValueError(f">>> !!! >>> eval_metric_name = {self.eval_metric_name}")
        self.logger.info(f">>> eval_metrics = {eval_metrics}")

        self.logger.info(f">>> Evaluation Task: {eval_task_name}")
        task_info = eval_task_obj.load_task()
        dataset_list = task_info["data"]

        re_number = re.compile(r"([-+]?[0-9]+)")  # Match integers
        re_boxed = re.compile(r"boxed{(.*?)}")  # Or r"\\boxed{(.*?)}" to match r"\boxed{}" answers
        # re_option = re.compile(r"([A-Z])|(\(A-Z\))|(A-Z\))")  # Match options: A, B, C, D, etc.

        def extract_math_answers(
                raw_pred_str: str
        ) -> List[str]:
            math_answers = set()  # To avoid duplication
            raw_pred_str = raw_pred_str.strip()

            boxed_answers = re.findall(re_boxed, raw_pred_str)
            if isinstance(boxed_answers, list) and len(boxed_answers) > 0:
                for boxed_ans in boxed_answers:
                    boxed_ans = str(boxed_ans).replace("\n", "").strip()
                    math_answers.add(boxed_ans)
                    math_answers.add(r"\boxed{" + boxed_ans + r"}")

            if "Final Answer:" in raw_pred_str:
                # Match the all integer numbers after "Final Answer:"
                final_pred_str = raw_pred_str.split("Final Answer:")[-1].strip()
                pred_answers = re.findall(re_number, final_pred_str)
                if len(pred_answers) > 0:
                    for answer_to_add in pred_answers:
                        if isinstance(answer_to_add, str) and len(answer_to_add) > 0:
                            math_answers.add(answer_to_add)
                            if answer_to_add.startswith("+") and len(answer_to_add) > 1:
                                math_answers.add(answer_to_add[1:])

                # Match the whole expression after "Final Answer:" and "answer is"
                if "answer is" in final_pred_str:  # Targeting the form "Final Answer: the final answer is xxx"
                    final_pred_str = final_pred_str.split("answer is")[-1].strip()
                if len(final_pred_str) > 0:
                    math_answers.add(final_pred_str)
                    final_pred_str = final_pred_str.replace(" ", "").strip()  # Spaces can be ignored
                    if len(final_pred_str) > 0:
                        math_answers.add(final_pred_str)
                    if "$" in final_pred_str:  # the math environment symbol "$" can be ignored
                        final_pred_str = final_pred_str.replace("$", "").strip()
                        if len(final_pred_str) > 0:
                            math_answers.add(final_pred_str)

            math_answers = list(math_answers)
            math_answers.sort()
            return math_answers

        # Deal with each task (and sub-tasks)
        all_scores = dict()  # The scores of the whole task/dataset (all subtasks/subsets)
        all_score_values = dict()
        data_item_cnt_total = 0
        miss_final_cnt_total = 0
        end_with_eot_cnt_total = 0
        show_cnt = 100
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
            if self.verbose:
                self.logger.info(f">>> [Dataset: {ds_id}] [Eval: {eval_split}] # = {len_dataset}")

            assert ds_id in gen_results
            cur_results = gen_results[ds_id]
            assert isinstance(cur_results, list) and len(cur_results) == len_dataset > 0

            miss_final_cnt_ds = 0
            end_with_eot_cnt_ds = 0
            cur_ds_score_values = dict()  # The scores of the whole subtask/subset
            cur_ds_results = []
            for idx, cur_res_dict in enumerate(cur_results):
                # Load the attributes of the data item
                gen_prompt = str(cur_res_dict["prompt"]).strip()
                prediction = str(cur_res_dict["output_text"]).strip()  # model prediction to evaluate
                references = cur_res_dict["answers"]  # golden references (correct answers)
                references = [str(_ref).strip() for _ref in references]
                info = cur_res_dict["info"]  # task-specific information

                if "end_with_eot" in cur_res_dict:
                    end_with_eot = bool(cur_res_dict["end_with_eot"])  # True if the output ends with end-of-text
                    if end_with_eot:
                        end_with_eot_cnt_ds += 1
                        end_with_eot_cnt_total += 1

                # Extract the final answer from the generated output
                assert isinstance(info, dict) and "task_type" in info
                task_type = info["task_type"]
                miss_final = False  # Also, count the number of missing final answer.
                pred_final = ""
                match task_type:
                    case "math":
                        lang_code = "en"
                        if "Final Answer:" in prediction or r"\boxed{" in prediction:
                            # pred_final = prediction.split("Final Answer:")[-1].strip()
                            # Extract the integer numbers and the whole expression (after "Final Answer:")
                            pred_final = extract_math_answers(prediction)
                            if len(pred_final) == 0:
                                miss_final = True

                            # Similarly, deal with the math references
                            ref_clear = set()  # To avoid duplication
                            for ref in references:
                                ref = ref.strip()
                                if ref.isdecimal():
                                    ref_clear.add(ref)
                                    ref_clear.add(r"\boxed{" + ref + r"}")
                                    continue
                                ref = ref.replace(" ", "").strip()  # Spaces can be ignored
                                if len(ref) > 0:
                                    ref_clear.add(ref)
                                    ref_clear.add(r"\boxed{" + ref + r"}")
                                if "$" in ref:  # the math environment symbol "$" can be ignored
                                    ref = ref.replace("$", "").strip()
                                    if len(ref) > 0:
                                        ref_clear.add(ref)
                                        ref_clear.add(r"\boxed{" + ref + r"}")
                            references = list(ref_clear)
                            references.sort()
                        else:
                            pred_final = ""
                            miss_final = True
                    case "qa":
                        lang_code = "en"
                        if "Final Answer:" in prediction:
                            pred_final = prediction.split("Final Answer:")[-1].strip()
                        else:
                            miss_final = True
                    case "sum":
                        lang_code = "en"
                        if "Final Summary:" in prediction:
                            pred_final = prediction.split("Final Summary:")[-1].strip()
                        else:
                            miss_final = True
                    case _:
                        raise ValueError(f"ValueError: task_type = {task_type}")
                cur_res_dict["miss_final"] = miss_final

                cur_res_dict["eval_score"] = dict()
                if miss_final:
                    miss_final_cnt_ds += 1
                    miss_final_cnt_total += 1

                for cur_metric in eval_metrics:
                    if cur_metric != "mcqa" and miss_final:
                        continue  # Only "mcqa" metric will deal with cases without "Final Answer"
                    # Compute the evaluation score
                    assert cur_metric in self.metric_func
                    cur_metric_func = self.metric_func[cur_metric]

                    if isinstance(pred_final, str):
                        cur_score = cur_metric_func(
                            prediction=prediction, references=references, item_info=info,
                            gen_prompt=gen_prompt, pred_final=pred_final, lang_code=lang_code)
                    elif isinstance(pred_final, list):  # If we have multiple candidate predictions
                        assert len(pred_final) > 0
                        cur_score = cur_metric_func(
                            prediction=prediction, references=references, item_info=info,
                            gen_prompt=gen_prompt, pred_final=pred_final[0], lang_code=lang_code)
                        assert isinstance(cur_score, dict) and "score" in cur_score
                        _best_score_value = cur_score["score"]  # To pick the candidate with the best score

                        for _pred_final in pred_final[1:]:
                            assert isinstance(_pred_final, str)
                            _cur_score = cur_metric_func(
                                prediction=prediction, references=references, item_info=info,
                                gen_prompt=gen_prompt, pred_final=_pred_final, lang_code=lang_code)
                            assert isinstance(_cur_score, dict) and "score" in _cur_score
                            _cur_score_value = _cur_score["score"]
                            if _cur_score_value >= _best_score_value:
                                cur_score = _cur_score
                    else:
                        raise ValueError(f"ValueError: pred_final = {pred_final}")

                    # Save the score to the current data item dict
                    cur_res_dict["eval_score"][cur_metric] = cur_score
                    cur_res_dict["lang"] = lang_code
                    # Save the score to the dict of the whole subtask/subset
                    assert isinstance(cur_score, dict) and "score" in cur_score
                    cur_score_value = cur_score["score"]
                    if cur_metric not in cur_ds_score_values:
                        cur_ds_score_values[cur_metric] = [cur_score_value]
                    else:
                        cur_ds_score_values[cur_metric].append(cur_score_value)

                cur_ds_results.append(cur_res_dict)
                if self.verbose and len(cur_ds_results) % show_cnt == 0:
                    self.logger.info(f">>> Progress: [{ds_id}] [{len(cur_ds_results)} / {len_dataset}] "
                                     f"[end_with_eot_cnt_ds = {end_with_eot_cnt_ds}]"
                                     f"[miss_final_cnt_ds = {miss_final_cnt_ds}]")

            # The score statistics of the current subtask/subset
            ds_score_stat = dict()
            for metric_name, score_value_list in cur_ds_score_values.items():
                ds_num_items = len(score_value_list)
                ds_score_avg = float(np.mean(score_value_list).item())
                if metric_name == "mcqa":
                    ds_score_avg_rectify = ds_score_avg
                else:
                    ds_score_avg_rectify = sum(score_value_list) / (len(score_value_list) + miss_final_cnt_ds)
                ds_score_stat[metric_name] = {
                    "num_items": ds_num_items,
                    "score_avg": ds_score_avg,
                    "score_avg_rectify": ds_score_avg_rectify,  # Treat the scores of missing-answer items as 0
                }
                self.logger.info(f">>> Subset Scores [{ds_id}] [Metric: {metric_name}]: "
                                 f"(num_items = {len(score_value_list)}) "
                                 f"score_avg = {ds_score_avg:.5f}; score_avg_rectify = {ds_score_avg_rectify:.5f}")

                if metric_name not in all_score_values:
                    all_score_values[metric_name] = score_value_list
                else:
                    all_score_values[metric_name].extend(score_value_list)

            all_scores[ds_id] = {
                "ds_results": cur_ds_results,
                "ds_scores": cur_ds_score_values,
                "ds_score_stat": ds_score_stat,
            }
            data_item_cnt_ds = len(cur_ds_results)
            data_item_cnt_total += data_item_cnt_ds
            if self.verbose:
                self.logger.info(f">>> Done Subtask/Subset. [{ds_id}] "
                                 f"[data_item_cnt_ds = {data_item_cnt_ds}] "
                                 f"[end_with_eot_cnt_ds = {end_with_eot_cnt_ds}] "
                                 f"[miss_final_cnt_ds = {miss_final_cnt_ds}]\n")

        # Compute the overall score statistics of different metrics and show stats
        all_score_stat = dict()
        for metric_name, score_value_list in all_score_values.items():
            # The score statistics of the whole task/dataset
            num_items = len(score_value_list)
            score_avg = float(np.mean(score_value_list).item())
            if metric_name == "mcqa":
                score_avg_rectify = score_avg
            else:
                score_avg_rectify = sum(score_value_list) / (len(score_value_list) + miss_final_cnt_total)
            all_score_stat[metric_name] = {
                "num_items": num_items,
                "score_avg": score_avg,
                "score_avg_rectify": score_avg_rectify,  # Treat the scores of missing-answer items as 0
            }
            self.logger.info(f">>> Overall Scores [Metric: {metric_name}]: (num_items = {num_items}) "
                             f"score_avg = {score_avg:.5f}; score_avg_rectify = {score_avg_rectify:.5f}")
        all_scores["all_score_stat"] = all_score_stat
        self.logger.info(f">>> DONE ALL. [Task: {eval_task_name}] "
                         f"[# Data items in total = {data_item_cnt_total}] "
                         f"[# Items end with eot = {end_with_eot_cnt_total}] "
                         f"[# Effective data items = {data_item_cnt_total - miss_final_cnt_total}] "
                         f"[# Missing Final Answer = {miss_final_cnt_total}]\n")

        # Save the generation outputs
        dumped = json.dumps(
            all_scores,
            indent=2,  # indent=None,
            default=self._handle_non_serializable,
            ensure_ascii=True,
        )
        with open(output_eval_fp, "w", encoding="utf-8") as fp_out:
            fp_out.write(dumped)
        self.logger.info(
            f">>> hf_id = {self.hf_id}; model_path = {self.model_path}\n"
            f"output_dir: {output_dir}"
        )


def main(
    task: int = 0,
    eval_task_name="",
    eval_metric_name="ALL",
    hf_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    seed: int = 42,
    cuda: Optional[str] = None,
    bsz: int = 1,
    verbose: bool = False,
    debug: bool = False,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    **kwargs
) -> None:
    """
    [Stage 2: Answer Extraction & Evaluation]
    Run LLM evaluation using different auto-eval metrics.

    :param task: 1. language model evaluation.
    :param eval_task_name: The name(s) of the evaluation task. (e.g., "xsum", "xlsum", and "gsm8k,math500")
    :param eval_metric_name: The name(s) of the evaluation metric. (e.g., "rouge", "bertscore", and "rouge,bertscore"). Default: None -- Use all default metrics for the task.
    :param hf_id: ORGANIZATION_NAME/MODEL_NAME, e.g., "meta-llama/Llama-3.1-8B-Instruct"
    :param cache_dir: The root directory of the cache.
    :param project_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param bsz: The batch size.
    :param verbose: Verbose mode: show logs.
    :param debug: Debugging / developing mode.
    :param output_dir: The path to the output file where the result metrics will be saved.
    :param overwrite: Overwrite existing output files.
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("LM_Eval")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Unused parameters in kwargs: {kwargs}")
    logger.info(f">>> cuda_dict: {cuda_dict}")

    if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
        os.environ["HF_HOME"] = cache_dir
        # os.environ["HF_HOME"] = os.path.join(cache_dir, "datasets")
        # os.environ["HF_HOME"] = os.path.join(cache_dir, "hub")
        nltk_download_dir = os.path.join(cache_dir, "downloads/nltk_data")
    else:
        cache_dir = None
        nltk_download_dir = os.path.join("downloads/nltk_data")

    # nltk packages for Meteor
    os.makedirs(nltk_download_dir, exist_ok=True)
    nltk.data.path.append(nltk_download_dir)
    try:
        nltk.download("wordnet", download_dir=nltk_download_dir)
        nltk.download("punkt_tab", download_dir=nltk_download_dir)
        nltk.download("omw-1.4", download_dir=nltk_download_dir)
    except Exception as e:
        if verbose:
            logger.info(e)

    if eval_task_name == "MATH_ALL":
        eval_task_name = ["gsm8k", "gsm8k_platinum", "math500", "amc23", "aime24", "aime25"]
    elif eval_task_name == "Competition":
        eval_task_name = ["amc23", "aime24", "aime25"]
    else:
        eval_task_name = str(eval_task_name).strip()

    lm_eval = LMEval(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        eval_task_name=eval_task_name,
        eval_metric_name=eval_metric_name,
        cache_dir=cache_dir,
        project_dir=project_dir,
        hf_id=hf_id,
        bsz=max(int(bsz), 1),
        debug=debug,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    task = int(task)
    match task:
        case 1:
            if isinstance(eval_task_name, tuple) or isinstance(eval_task_name, list):
                for cur_task_name in eval_task_name:
                    assert cur_task_name in lm_eval.task_class_dict, \
                        f"AssertionError: task name {cur_task_name} not in task_class_dict"
                    cur_task_name = str(cur_task_name).strip()
                    logger.info(f">>> <START> {cur_task_name}\n")
                    lm_eval.lm_evaluate(eval_task_name=cur_task_name)
                    logger.info(f">>> <END> {cur_task_name}\n\n\n")
            elif isinstance(eval_task_name, str):
                assert eval_task_name in lm_eval.task_class_dict, \
                    f"AssertionError: task name {eval_task_name} not in task_class_dict"
                eval_task_name = str(eval_task_name).strip()
                logger.info(f">>> <START> {eval_task_name}\n")
                lm_eval.lm_evaluate(eval_task_name=eval_task_name)
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
