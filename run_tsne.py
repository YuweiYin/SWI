#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import re
import time
import random
from typing import Optional, List

import fire
import numpy as np
from nltk.corpus import wordnet

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from tasks.tasks_utils import *
from utils.init_functions import logger_setup, random_setup
from utils.data_io import DataIO


def main(
    task: int = 1,
    cache_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    seed: int = 42,
    verbose: bool = False,
    output_dir: Optional[str] = None,
    hf_id_generation: str = "meta-llama/Llama-3.1-8B-Instruct",
    hf_id_embedding: str = "Qwen/Qwen3-Embedding-8B",
    **kwargs
) -> None:
    """
    t-SNE visualization.

    :param task: 1. prepare for the text data; 2. prepare for the embeddings; 3. show intent distribution (t-SNE).
    :param cache_dir: The root directory of the cache.
    :param project_dir: The directory of the project root.
    :param seed: Random seed of all modules.
    :param verbose: Verbose mode: show logs.
    :param output_dir: The directory of the output results.
    :param hf_id_generation: The model used to generate outputs.
    :param hf_id_embedding: The embedding model.
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setups
    logger = logger_setup("Intent_Distribution")
    random_setup(seed=seed, has_cuda=False)

    if isinstance(kwargs, dict):
        logger.info(f">>> Extra parameters in kwargs: {kwargs}\n")

    # Set the tokenizer path
    os.environ["HF_HOME"] = cache_dir
    hf_name_embedding = "--".join(hf_id_embedding.split("/"))
    local_model_path = os.path.join(cache_dir, "models--" + hf_name_embedding, "snapshots/model")
    model_path = local_model_path if os.path.isdir(local_model_path) else hf_name_embedding

    task = int(task)
    match task:
        case 1:
            # Prepare for the text data
            assert os.path.isdir(output_dir), output_dir

            def get_samples(cur_task_name: str, num_fetch: int) -> List[str]:
                hf_name_gen = "--".join(hf_id_generation.split("/"))
                intent_ptn = re.compile(r"<INTENT>(.*?)</INTENT>")

                output_fp = os.path.join(output_dir, cur_task_name, hf_name_gen, "results_gen.json")
                cur_outputs = DataIO.load_json(output_fp, verbose=verbose)
                assert isinstance(cur_outputs, dict) and len(cur_outputs) > 0, type(cur_outputs)
                assert isinstance(num_fetch, int) and num_fetch > 0, num_fetch

                # Extract the intent statements
                _all_intents = []
                for ds_id, cur_results in cur_outputs.items():
                    assert isinstance(cur_results, list) and len(cur_results) > 0, type(cur_results)
                    if verbose:
                        logger.info(f">>> [Dataset: {ds_id}] [#Items = {len(cur_results)}]")

                    for cur_gen_output in cur_results:
                        assert isinstance(cur_gen_output, dict) and len(cur_gen_output) > 0, type(cur_gen_output)
                        output_text = cur_gen_output["output_text"]

                        assert isinstance(output_text, str)
                        cur_intents = re.findall(intent_ptn, output_text)
                        cur_intents = [_x for _x in cur_intents if len(_x) > 0]
                        if isinstance(cur_intents, list) and len(cur_intents) > 0:
                            _all_intents.extend(cur_intents)

                # Randomly sample `num_fetch` instances
                _intent_samples = random.sample(_all_intents, num_fetch)
                return _intent_samples

            # Obtain the text data (300 instances per Sum dataset; 500 instances per QA/Math dataset)
            all_intent_samples = {"sum": [], "qa": [], "math": []}
            if "use_intent_statement" in kwargs:  # randomly sample intent statements (full sentences "To do sth.")
                for task_name, task_class in SUM_CLASS_DICT.items():
                    all_intent_samples["sum"].extend(get_samples(task_name, num_fetch=300))
                for task_name, task_class in QA_CLASS_DICT.items():
                    all_intent_samples["qa"].extend(get_samples(task_name, num_fetch=500))
                for task_name, task_class in MATH_CLASS_DICT.items():
                    all_intent_samples["math"].extend(get_samples(task_name, num_fetch=500))
            else:
                # After the step 4 in `run_stat.py`, we can use the saved intent verbs (of top frequency)
                # for task_type in all_intent_samples.keys():
                for task_type in ["sum", "qa", "math"]:
                    intent_verbs_filepath = os.path.join(output_dir, f"intent_stat-{task_type}.json")
                    assert os.path.isfile(intent_verbs_filepath), intent_verbs_filepath
                    intent_verbs = DataIO.load_json(intent_verbs_filepath, verbose=verbose)
                    assert isinstance(intent_verbs, list) and len(intent_verbs) > 0, type(intent_verbs)
                    # `intent_verbs` is already sorted by the verb counts (without duplication)
                    if "num_intent_verbs" in kwargs:
                        intent_verbs = intent_verbs[:int(kwargs["num_intent_verbs"])]
                    all_intent_samples[task_type] = [str(_v["verb"]).strip() for _v in intent_verbs]

                # All English verbs
                english_verbs = set()
                for synset in wordnet.all_synsets(pos="v"):
                    for lemma in synset.lemmas():
                        english_verbs.add(str(lemma.name()).strip())
                all_intent_samples["english"] = list(english_verbs)

            # Save the text data
            intent_samples_fp = os.path.join(output_dir, "intent_samples_tsne.json")
            DataIO.save_json(intent_samples_fp, all_intent_samples, verbose=verbose)
        case 2:
            # Prepare for the embeddings
            intent_samples_fp = os.path.join(output_dir, "intent_samples_tsne.json")
            assert os.path.isfile(intent_samples_fp), intent_samples_fp
            all_intents = DataIO.load_json(intent_samples_fp, verbose=verbose)
            assert isinstance(all_intents, dict) and len(all_intents) > 0, type(all_intents)

            def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
                left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                if left_padding:
                    return last_hidden_states[:, -1]
                else:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = last_hidden_states.shape[0]
                    return last_hidden_states[
                        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

            # Load the embedding model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, padding_side="left", truncation_side="left", cache_dir=cache_dir,
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModel.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, cache_dir=cache_dir,
            )  # attn_implementation="flash_attention_2"

            all_embed = {"sum": [], "qa": [], "math": [], "english": []}
            embed_size = 4096
            done_cnt = 0
            show_cnt = 1000
            for task_type, intent_samples in all_intents.items():
                assert isinstance(task_type, str) and isinstance(intent_samples, list)
                if len(intent_samples) == 0:
                    logger.info(f">>> !!! >>> [task_type: {task_type}] intent_samples is empty!")
                    continue

                for idx, intent in enumerate(intent_samples):
                    assert isinstance(intent, str) and len(intent) > 0, intent

                    # Tokenize the intent
                    batch_dict = tokenizer(
                        [intent],
                        padding=True,
                        truncation=True,
                        max_length=embed_size,  # 8192
                        return_tensors="pt",
                    )  # batch size = 1
                    batch_dict.to(model.device)

                    # Run the model and obtain the embeddings
                    outputs = model(**batch_dict)
                    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                    if "do_normalize" in kwargs:
                        embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize the embeddings
                    embeddings = embeddings.cpu().detach()
                    all_embed[task_type].append(embeddings)  # The shape of embeddings: [bsz, embed_size]

                    done_cnt += 1
                    if done_cnt % show_cnt == 0:
                        logger.info(f">>> >>> done_cnt = {done_cnt}")

                all_embed[task_type] = torch.cat(all_embed[task_type], dim=0).tolist()

            # Save the embeddings
            if "do_normalize" in kwargs:
                intent_embeddings_fp = os.path.join(
                    output_dir, f"intent_embeddings_tsne-normalize---{hf_name_embedding}.json")
            else:
                intent_embeddings_fp = os.path.join(
                    output_dir, f"intent_embeddings_tsne---{hf_name_embedding}.json")
            DataIO.save_json(intent_embeddings_fp, all_embed, verbose=verbose)
        case 3:
            # Plot intent distribution (t-SNE)
            if "do_normalize" in kwargs:
                intent_embeddings_fp = os.path.join(
                    output_dir, f"intent_embeddings_tsne-normalize---{hf_name_embedding}.json")
            else:
                intent_embeddings_fp = os.path.join(
                    output_dir, f"intent_embeddings_tsne---{hf_name_embedding}.json")
            assert os.path.isfile(intent_embeddings_fp), intent_embeddings_fp
            all_embed = DataIO.load_json(intent_embeddings_fp, verbose=verbose)
            assert isinstance(all_embed, dict) and len(all_embed) > 0, type(all_embed)
            assert "sum" in all_embed and "qa" in all_embed and "math" in all_embed
            len_sum, len_qa, len_math = len(all_embed["sum"]), len(all_embed["qa"]), len(all_embed["math"])
            len_eng = len(all_embed["english"])

            # Initialize the t-SNE model
            tsne = TSNE(
                n_components=2, perplexity=30, max_iter=1000, n_iter_without_progress=300, random_state=seed,
                init="pca", learning_rate="auto", method="barnes_hut",
                metric="euclidean",  # the default metric is "euclidean"
                # metric="cosine",
            )

            # plt.figure(figsize=(10, 8))
            fig, ax = plt.subplots(figsize=(5, 6), nrows=1, ncols=1)  # ncols=2
            fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9, bottom=0.25)

            colors_sum, colors_qa, colors_math = ["coral"] * len_sum, ["darkviolet"] * len_qa, ["limegreen"] * len_math
            assert "draw_option" in kwargs, kwargs
            draw_option = str(kwargs["draw_option"]).strip()
            dot_size = 15
            alpha = 0.6
            match draw_option:
                # Fit the t-SNE model and draw scatter plots
                case "sum":
                    embed_sum = np.array(all_embed["sum"], dtype=np.float32)
                    embed_2d_sum = tsne.fit_transform(embed_sum)
                    plt.scatter(embed_2d_sum[:, 0], embed_2d_sum[:, 1], c=colors_sum, alpha=alpha)
                    ax.legend("SUM", loc=(0.03, 0.78), labelspacing=0.1, fontsize=14)
                case "qa":
                    embed_qa = np.array(all_embed["qa"], dtype=np.float32)
                    embed_2d_qa = tsne.fit_transform(embed_qa)
                    plt.scatter(embed_2d_qa[:, 0], embed_2d_qa[:, 1], c=colors_qa, alpha=alpha)
                    ax.legend("QA", loc=(0.03, 0.78), labelspacing=0.1, fontsize=14)
                case "math":
                    embed_math = np.array(all_embed["math"], dtype=np.float32)
                    embed_2d_math = tsne.fit_transform(embed_math)
                    plt.scatter(embed_2d_math[:, 0], embed_2d_math[:, 1], c=colors_math, alpha=alpha)
                    ax.legend("MATH", loc=(0.03, 0.78), labelspacing=0.1, fontsize=14)
                case "sum_qa":
                    embed_sum_qa = np.array(all_embed["sum"] + all_embed["qa"], dtype=np.float32)
                    embed_2d_sum_qa = tsne.fit_transform(embed_sum_qa)
                    plt.scatter(embed_2d_sum_qa[:len_sum, 0], embed_2d_sum_qa[:len_sum, 1],
                                c=colors_sum, s=dot_size, alpha=alpha)
                    plt.scatter(embed_2d_sum_qa[len_sum:, 0], embed_2d_sum_qa[len_sum:, 1],
                                c=colors_qa, s=dot_size, alpha=alpha)
                    ax.legend(("SUM", "QA"), loc=(0.03, 0.78), labelspacing=0.1, fontsize=14)
                case "sum_math":
                    embed_sum_math = np.array(all_embed["sum"] + all_embed["math"], dtype=np.float32)
                    embed_2d_sum_math = tsne.fit_transform(embed_sum_math)
                    plt.scatter(embed_2d_sum_math[:len_sum, 0], embed_2d_sum_math[:len_sum, 1],
                                c=colors_sum, s=dot_size, alpha=alpha)
                    plt.scatter(embed_2d_sum_math[len_sum:, 0], embed_2d_sum_math[len_sum:, 1],
                                c=colors_math, s=dot_size, alpha=alpha)
                    ax.legend(("SUM", "MATH"), loc=(0.03, 0.78), labelspacing=0.1, fontsize=14)
                case "qa_math":
                    embed_qa_math = np.array(all_embed["qa"] + all_embed["math"], dtype=np.float32)
                    embed_2d_qa_math = tsne.fit_transform(embed_qa_math)
                    plt.scatter(embed_2d_qa_math[:len_qa, 0], embed_2d_qa_math[:len_qa, 1],
                                c=colors_qa, s=dot_size, alpha=alpha)
                    plt.scatter(embed_2d_qa_math[len_qa:, 0], embed_2d_qa_math[len_qa:, 1],
                                c=colors_math, s=dot_size, alpha=alpha)
                    ax.legend(("QA", "MATH"), loc=(0.03, 0.78), labelspacing=0.1, fontsize=14)
                case "all_vs_eng":
                    embed_all_vs_eng = np.array(
                        all_embed["sum"] + all_embed["qa"] + all_embed["math"] + all_embed["english"], dtype=np.float32)
                    embed_2d_all_vs_eng = tsne.fit_transform(embed_all_vs_eng)
                    plt.scatter(embed_2d_all_vs_eng[(len_sum + len_qa + len_math):, 0],
                                embed_2d_all_vs_eng[(len_sum + len_qa + len_math):, 1],
                                c=["cornflowerblue"] * len_eng, s=dot_size, alpha=alpha)
                    plt.scatter(embed_2d_all_vs_eng[:(len_sum + len_qa + len_math), 0],
                                embed_2d_all_vs_eng[:(len_sum + len_qa + len_math), 1],
                                c=["tomato"] * (len_sum + len_qa + len_math), s=dot_size, alpha=alpha)
                    ax.legend(("English verbs", "Intent verbs"), loc=(0.03, 0.83), labelspacing=0.1, fontsize=14)

                    # fig.text(0.5, 0.92, "Intents among English verbs",
                    #          horizontalalignment="center", color="black", weight="bold", size=20)
                    fig.text(0.52, 0.19, "(b)",
                             horizontalalignment="center", color="black", weight="bold", size=24)  # size="large"
                case _:
                    embed_all = np.array(all_embed["sum"] + all_embed["qa"] + all_embed["math"], dtype=np.float32)
                    embed_2d_all = tsne.fit_transform(embed_all)
                    plt.scatter(embed_2d_all[:len_sum, 0], embed_2d_all[:len_sum, 1],
                                c=colors_sum, s=dot_size, alpha=alpha)
                    plt.scatter(embed_2d_all[len_sum:(len_sum + len_qa), 0],
                                embed_2d_all[len_sum:(len_sum + len_qa), 1],
                                c=colors_qa, s=dot_size, alpha=alpha)
                    plt.scatter(embed_2d_all[(len_sum + len_qa):(len_sum + len_qa + len_math), 0],
                                embed_2d_all[(len_sum + len_qa):(len_sum + len_qa + len_math), 1],
                                c=colors_math, s=dot_size, alpha=alpha)
                    ax.legend(("SUM", "QA", "MATH"), loc=(0.03, 0.78), labelspacing=0.1, fontsize=14)

                    # fig.text(0.5, 0.92, "Intents across Tasks",
                    #          horizontalalignment="center", color="black", weight="bold", size=20)
                    fig.text(0.52, 0.19, "(a)",
                             horizontalalignment="center", color="black", weight="bold", size=24)  # size="large"

            # plt.axis("off")
            plt.xticks([])  # Removes x-axis ticks and labels
            plt.yticks([])  # Removes y-axis ticks and labels

            if "do_save" in kwargs:
                save_fp = os.path.join(
                    project_dir, "figures", f"_tsne_intent_distribution---{hf_name_embedding}---{draw_option}.pdf")
                plt.savefig(save_fp, format="pdf", dpi=600)
            else:
                plt.show()
        case _:
            raise ValueError(f"ValueError: task = {task}")

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
