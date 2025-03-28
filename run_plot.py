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
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from utils.init_functions import logger_setup, random_setup


def radar_factory(
        num_vars,
        frame="circle",
):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {"circle", "polygon"}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's auto-conversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self.close_line(line)

        @staticmethod
        def close_line(line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        @staticmethod
        def gen_axes_patch():
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError(f"Unknown value for \"frame\": {frame}")

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be "left"/"right"/"top"/"bottom"/"circle".
                spine = Spine(axes=self,
                              spine_type="circle",
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {"polar": spine}
            else:
                raise ValueError(f"Unknown value for \"frame\": {frame}")

    register_projection(RadarAxes)
    return theta


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

        # self.SMALL_SIZE = 8
        # self.MEDIUM_SIZE = 10
        # self.BIGGER_SIZE = 12

        # plt.rc("font",**{"family": "sans-serif", "sans-serif": ["Helvetica"]})
        # plt.rc("font", **{"family": "serif", "serif": ["Times"]})
        plt.rcParams.update({
            # "text.usetex": True,
            "font.family": "Times New Roman"
        })
        # plt.rc("xtick", labelsize=20)
        # plt.rc("ytick", labelsize=20)

    def plot_radar_chart(self):
        plt.rc("xtick", labelsize=28)
        plt.rc("ytick", labelsize=28)

        # Mathematical Reasoning: GSM8K, GSM8K-Platinum, MATH500
        # Multiple-Choice Science QA: ARC (AI2 Reasoning Challenge), BBH (BIG-Bench Hard), MMLU, MMLU-Pro
        # Summarization (avg of all the six datasets)
        data = [
            # Evaluation Results
            ("Main Experiments", [
                [81.14, 78.47, 36.80, 25.00, 39.27, 52.40, 56.65, 76.44],  # Baseline
                [84.20, 81.20, 41.80, 30.00, 45.16, 62.19, 64.29, 87.82],  # SWI
            ]),
        ]

        spoke_labels = ["GSM8K-Platinum", "GSM8K", "MATH500", "AMC23", "MMLU-Pro", "MMLU", "BBH", "ARC"]
        N = len(spoke_labels)
        theta = radar_factory(N, frame="polygon")

        fig, ax = plt.subplots(
            figsize=(9, 9), nrows=1, ncols=1,  # ncols=2
            subplot_kw=dict(projection="radar")
        )
        fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.85, bottom=0.1)

        # Plot the data on separate Axes
        colors = ["cornflowerblue", "coral"]
        ax.set_rgrids([0.0, 0.2, 0.4, 0.6, 0.8])
        title, case_data = data[0]
        for d, color in zip(case_data, colors):  # Draw each polygon
            d = [number / 100.0 for number in d]
            ax.plot(theta, d, color=color, linewidth=2)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label="_nolegend_")
        ax.set_varlabels(spoke_labels)
        ax.set_ylim(0.1, 0.9)

        # add legend relative to top-left plot
        labels = ("Baseline", "SWI")
        ax.legend(
            labels, loc=(0.74, 0.93), labelspacing=0.1, fontsize=24)

        fig.text(0.52, 0.02, "(c)",
                 horizontalalignment="center", color="black", weight="bold", size=36)  # size="large"

        if self.do_save:
            save_fp = os.path.join(self.save_dir, "_radar_chart.pdf")
            plt.savefig(save_fp, format=self.save_format, dpi=600)
        else:
            plt.show()

    def plot_bar_chart(self):
        plt.rc("xtick", labelsize=16)
        plt.rc("ytick", labelsize=16)

        data = [
            # Evaluation Results
            ("Main Experiments", {
                "Baseline": (78.5, 81.1, 36.8, 25.0, 76.4, 56.7, 52.4, 39.3),  # Baseline
                "SWI": (81.2, 84.2, 41.8, 30.0, 87.8, 64.3, 62.2, 45.2),  # SWI
            }),
        ]

        x_labels = ["GSM8K", "GSM8K-P", "MATH500", "AMC23", "ARC", "BBH", "MMLU", "MMLU-Pro"]

        fig, ax = plt.subplots(
            figsize=(6, 6), nrows=1, ncols=1,
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
            rects = ax.bar(x + offset, measurement, width, label=attribute,
                           color=colors[color_idx], alpha=0.8)  # hatch="//"
            # ax.bar_label(rects, padding=3)  # Show the value of each bar
            # ax.plot(x + offset, measurement, marker="o", color=colors[color_idx])
            multiplier += 1
            color_idx += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Performance (EM/Acc %)", fontsize=14)
        ax.set_xticks(x + width, x_labels, rotation=60)
        # ax.legend(loc="upper left", ncols=3)
        ax.set_ylim(0, 100)
        ax.grid(axis="y")  # ax.grid()

        # Add legend relative to top-left plot
        labels = ("Baseline", "SWI")
        ax.legend(labels, loc=(0.68, 0.83), labelspacing=0.1, fontsize=14)

        fig.text(0.5, 0.92, "Improvement brought by SWI",
                 horizontalalignment="center", color="black", weight="bold", size=20)
        fig.text(0.52, 0.05, "(c)",
                 horizontalalignment="center", color="black", weight="bold", size=24)  # size="large"

        if self.do_save:
            save_fp = os.path.join(self.save_dir, "_bar_chart.pdf")
            plt.savefig(save_fp, format=self.save_format, dpi=600)
        else:
            plt.show()


def main(
        task: int = 0,
        seed: int = 42,
        verbose: bool = False,
        do_save: bool = False,
        save_format: str = "pdf",
        **kwargs
) -> None:
    """
    Run plotting.

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
            plotter.plot_radar_chart()
            plotter.plot_bar_chart()
        case _:
            raise ValueError(f"ValueError: task = {task}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
