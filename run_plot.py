#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import time

import fire
import numpy as np
import matplotlib.pyplot as plt

from utils.init_functions import logger_setup, random_setup


class Plotter:

    def __init__(
            self,
            verbose: bool,
            logger,
            seed: int = 42,
            do_save: bool = False,
            save_format: str = "pdf",
    ):
        self.verbose = verbose
        self.logger = logger
        self.seed = seed
        self.do_save = do_save
        self.save_format = save_format
        assert self.save_format in ["pdf", "png"]

        self.save_dir = "figures"
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        plt.rcParams.update({
            # "text.usetex": True,
            "font.family": "Times New Roman"
        })

    def plot_bar_chart_overview(self):
        plt.rc("xtick", labelsize=16)
        plt.rc("ytick", labelsize=16)

        data = [
            # Score lists
            ("Main Experiments", {
                "w/o SWI": (11.29, 16.92, 15.01, 56.65, 52.40, 38.20),  # Baseline
                "w/ SWI": (13.80, 19.57, 16.53, 63.11, 59.22, 43.00),  # SWI
            }),  # LLaMA-3.1-8B Results
        ]

        x_labels = ["XL-Sum", "DialogSum", "WikiLingua", "BBH", "MMLU", "MATH500"]

        fig, ax = plt.subplots(
            figsize=(5, 6), nrows=1, ncols=1,  # ncols=2
        )
        fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9, bottom=0.25)

        # Plot the data on separate Axes
        colors = ["cornflowerblue", "coral"]
        title, case_data = data[0]

        x = np.arange(len(x_labels))  # the label locations
        width = 0.3  # the width of the bars
        multiplier = 0
        color_idx = 0
        for attribute, measurement in case_data.items():
            offset = width * multiplier
            ax.bar(x + offset, measurement, width, label=attribute, color=colors[color_idx], alpha=0.8)  # hatch="//"
            multiplier += 1
            color_idx += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Performance (ROUGE/Acc/EM %)", fontsize=14)
        ax.set_xticks(x + width, x_labels, rotation=60)
        ax.set_ylim(0, 65)
        ax.grid(axis="y")  # ax.grid()

        # Add legend relative to top-left plot
        labels = ("w/o SWI", "w/ SWI")
        ax.legend(labels, loc=(0.05, 0.777), labelspacing=0.1, fontsize=14)

        fig.text(0.5, 0.92, "Improvement brought by SWI",  # "Improvement by SWI over Baseline"
                 horizontalalignment="center", color="black", weight="bold", size=20)
        fig.text(0.52, 0.05, "(b)",
                 horizontalalignment="center", color="black", weight="bold", size=24)  # size="large"

        if self.do_save:
            save_fp = os.path.join(self.save_dir, "_bar_chart_overview.pdf")
            plt.savefig(save_fp, format=self.save_format, dpi=600)
        else:
            plt.show()

    def plot_bar_chart_intent_stat_sum(self):
        plt.rc("xtick", labelsize=16)
        plt.rc("ytick", labelsize=16)

        data = [
            # Score lists
            ("Intent Statistics", {
                # y: the number of each top frequent intent verb
                "# of intent verbs": (61918, 25172, 18999, 13570, 9645, 9250, 9048, 4955, 4445, 2656),
            }),  # LLaMA-3.1-8B Results
        ]

        # x: the top-10 intent verbs in this task (across multiple datasets)
        x_labels = ["provide", "highlight", "explain", "describe", "discuss",
                    "mention", "summarize", "state", "outline", "clarify"]

        # fig, ax = plt.subplots(layout="constrained")
        fig, ax = plt.subplots(
            figsize=(5, 6), nrows=1, ncols=1,  # ncols=2
        )
        fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9, bottom=0.25)

        # Plot the data on separate Axes
        colors = ["coral"]
        title, case_data = data[0]

        num_total_intent_verbs = 177085.0  # the total number of verbs in this task
        x = np.arange(len(x_labels))  # the label locations
        width = 0.5  # the width of the bars
        multiplier = 0
        color_idx = 0
        for attribute, measurement in case_data.items():
            offset = width
            measurement = [number * 100.0 / num_total_intent_verbs for number in measurement]
            ax.bar(x + offset, measurement, width, label=attribute, color=colors[color_idx], alpha=0.8)  # hatch="//"
            multiplier += 1
            color_idx += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Ratio (%)", fontsize=14)
        ax.set_xticks(x + width, x_labels, rotation=60)
        ax.grid(axis="y")  # ax.grid()

        fig.text(0.5, 0.92, "Top 10 Intent Verbs (Sum)",  # "Improvement by SWI over Baseline"
                 horizontalalignment="center", color="black", weight="bold", size=20)
        fig.text(0.50, 0.05, "(a)",
                 horizontalalignment="center", color="black", weight="bold", size=24)  # size="large"

        if self.do_save:
            save_fp = os.path.join(self.save_dir, "_bar_chart_intent_stat_sum.pdf")
            plt.savefig(save_fp, format=self.save_format, dpi=600)
        else:
            plt.show()

    def plot_bar_chart_intent_stat_qa(self):
        plt.rc("xtick", labelsize=16)
        plt.rc("ytick", labelsize=16)

        data = [
            # Score lists
            ("Intent Statistics", {
                # y: the number of each top frequent intent verb
                "# of intent verbs": (11733, 6758, 6239, 5316, 3169, 3030, 2970, 1473, 904, 813),
            }),  # LLaMA-3.1-8B Results
        ]

        # x: the top-10 intent verbs in this task (across multiple datasets)
        x_labels = ["identify", "select", "calculate", "provide", "determine",
                    "evaluate", "analyze", "find", "explain", "compare"]

        # fig, ax = plt.subplots(layout="constrained")
        fig, ax = plt.subplots(
            figsize=(5, 6), nrows=1, ncols=1,  # ncols=2
        )
        fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9, bottom=0.25)

        # Plot the data on separate Axes
        colors = ["cornflowerblue"]
        title, case_data = data[0]

        num_total_intent_verbs = 48721.0  # the total number of verbs in this task
        x = np.arange(len(x_labels))  # the label locations
        width = 0.5  # the width of the bars
        multiplier = 0
        color_idx = 0
        for attribute, measurement in case_data.items():
            offset = width
            measurement = [number * 100.0 / num_total_intent_verbs for number in measurement]
            ax.bar(x + offset, measurement, width, label=attribute, color=colors[color_idx], alpha=0.8)  # hatch="//"
            multiplier += 1
            color_idx += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Ratio (%)", fontsize=14)
        ax.set_xticks(x + width, x_labels, rotation=60)
        ax.grid(axis="y")  # ax.grid()

        fig.text(0.5, 0.92, "Top 10 Intent Verbs (QA)",  # "Improvement by SWI over Baseline"
                 horizontalalignment="center", color="black", weight="bold", size=20)
        fig.text(0.50, 0.05, "(b)",
                 horizontalalignment="center", color="black", weight="bold", size=24)  # size="large"

        if self.do_save:
            save_fp = os.path.join(self.save_dir, "_bar_chart_intent_stat_qa.pdf")
            plt.savefig(save_fp, format=self.save_format, dpi=600)
        else:
            plt.show()

    def plot_bar_chart_intent_stat_math(self):
        plt.rc("xtick", labelsize=16)
        plt.rc("ytick", labelsize=16)

        data = [
            # Score lists
            ("Intent Statistics", {
                # y: the number of each top frequent intent verb
                "# of intent verbs": (5043, 2550, 978, 639, 587, 387, 366, 272, 262, 247),
            }),  # LLaMA-3.1-8B Results
        ]

        # x: the top-10 intent verbs in this task (across multiple datasets)
        x_labels = ["calculate", "find", "determine", "add", "simplify",
                    "solve", "multiply", "express", "identify", "subtract"]

        # fig, ax = plt.subplots(layout="constrained")
        fig, ax = plt.subplots(
            figsize=(5, 6), nrows=1, ncols=1,  # ncols=2
        )
        fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9, bottom=0.25)

        # Plot the data on separate Axes
        colors = ["seagreen"]
        title, case_data = data[0]

        num_total_intent_verbs = 13575.0  # the total number of verbs in this task
        x = np.arange(len(x_labels))  # the label locations
        width = 0.5  # the width of the bars
        multiplier = 0
        color_idx = 0
        for attribute, measurement in case_data.items():
            offset = width
            measurement = [number * 100.0 / num_total_intent_verbs for number in measurement]
            ax.bar(x + offset, measurement, width, label=attribute, color=colors[color_idx], alpha=0.8)  # hatch="//"
            multiplier += 1
            color_idx += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Ratio (%)", fontsize=14)
        ax.set_xticks(x + width, x_labels, rotation=60)
        ax.grid(axis="y")  # ax.grid()

        fig.text(0.5, 0.92, "Top 10 Intent Verbs (Math)",  # "Improvement by SWI over Baseline"
                 horizontalalignment="center", color="black", weight="bold", size=20)
        fig.text(0.50, 0.05, "(c)",
                 horizontalalignment="center", color="black", weight="bold", size=24)  # size="large"

        if self.do_save:
            save_fp = os.path.join(self.save_dir, "_bar_chart_intent_stat_math.pdf")
            plt.savefig(save_fp, format=self.save_format, dpi=600)
        else:
            plt.show()


def main(
        task: int = 1,
        seed: int = 42,
        verbose: bool = False,
        do_save: bool = False,
        save_format: str = "pdf",
        **kwargs
) -> None:
    """
    Plot figures.

    :param task: 1. run plotting.
    :param seed: Random seed of all modules.
    :param verbose: Verbose mode: show logs.
    :param do_save: True if we want to save the figures.
    :param save_format: The format of the saved figures.
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("Plot")
    random_setup(seed=seed, has_cuda=False)

    if isinstance(kwargs, dict):
        logger.info(f">>> Unused parameters in kwargs: {kwargs}\n")

    plotter = Plotter(
        verbose=verbose,
        logger=logger,
        seed=seed,
        do_save=do_save,
        save_format=save_format,
    )

    task = int(task)
    match task:
        case 1:
            plotter.plot_bar_chart_overview()
            plotter.plot_bar_chart_intent_stat_sum()
            plotter.plot_bar_chart_intent_stat_qa()
            plotter.plot_bar_chart_intent_stat_math()
        case _:
            raise ValueError(f"ValueError: task = {task}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
