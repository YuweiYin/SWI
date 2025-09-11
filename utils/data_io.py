# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import json
import logging
from typing import Optional, Union
import numpy as np


class DataIO:

    def __init__(self):
        pass

    @staticmethod
    def handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    @staticmethod
    def load_json(
            filepath: str,
            mode: str = "r",
            encoding: str = "utf-8",
            verbose: bool = False,
    ) -> Union[dict, list]:
        results = []
        if os.path.isfile(filepath):
            if verbose:
                logging.info(f">>> [load_json] {filepath}")
            with open(filepath, mode, encoding=encoding) as fp_in:
                results = json.load(fp_in)
        else:
            if verbose:
                logging.info(f">>> [load_json] filepath does not exist: {filepath}")
        return results

    @staticmethod
    def save_json(
            filepath: str,
            data,
            mode: str = "w",
            encoding: str = "utf-8",
            indent: Optional[int] = None,
            verbose: bool = False,
    ) -> None:
        if verbose:
            logging.info(f">>> [save_json] {filepath}")
        with open(filepath, mode, encoding=encoding) as fp_out:
            json.dump(data, fp_out, indent=indent, ensure_ascii=True, default=DataIO.handle_non_serializable)

    @staticmethod
    def load_jsonl(
            filepath: str,
            mode: str = "r",
            encoding: str = "utf-8",
            verbose: bool = False,
    ) -> list:
        results = []
        if os.path.isfile(filepath):
            if verbose:
                logging.info(f">>> [load_jsonl] {filepath}")
            with open(filepath, mode, encoding=encoding) as fp_in:
                for line in fp_in:
                    results.append(json.loads(line))
        else:
            if verbose:
                logging.info(f">>> [load_jsonl] filepath does not exist: {filepath}")
        return results

    @staticmethod
    def save_jsonl(
            filepath: str,
            data: list,
            mode: str = "w",
            encoding: str = "utf-8",
            verbose: bool = False,
    ) -> None:
        if verbose:
            logging.info(f">>> [save_jsonl] {filepath}")
        with open(filepath, mode, encoding=encoding) as fp_out:
            for data_item in data:
                fp_out.write(json.dumps(data_item) + "\n")

    @staticmethod
    def show_dict(
            input_dict: dict,
            dict_name: str = "",
            logger=None,
    ):
        if logger is None:
            logger = logging.getLogger("DataIO")
        if dict_name == "":
            dict_name = "show_dict"

        if isinstance(input_dict, dict) and len(input_dict) > 0:
            for k, v in input_dict.items():
                logger.info(f">>> >>> [{dict_name}] {k}: {v}")
        else:
            logger.info(f">>> >>> [show_dict] input_dict is not dict or is empty.")
