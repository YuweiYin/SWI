from __future__ import annotations

import csv
import logging
import os

from scipy.stats import pearsonr, spearmanr

from evaluate_metrics.sentence_transformers import InputExample

logger = logging.getLogger(__name__)


class CECorrelationEvaluator:
    """
    This evaluator can be used with the CrossEncoder class. Given sentence pairs and continuous scores,
    it compute the pearson & spearman correlation between the predicted score for the sentence pair
    and the gold score.
    """

    def __init__(self, sentence_pairs: list[list[str]], scores: list[float], name: str = "", write_csv: bool = True):
        self.sentence_pairs = sentence_pairs
        self.scores = scores
        self.name = name

        self.csv_file = "CECorrelationEvaluator" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Pearson_Correlation", "Spearman_Correlation"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        sentence_pairs = []
        scores = []

        for example in examples:
            sentence_pairs.append(example.texts)
            scores.append(example.label)
        return cls(sentence_pairs, scores, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("CECorrelationEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)

        eval_pearson, _ = pearsonr(self.scores, pred_scores)
        eval_spearman, _ = spearmanr(self.scores, pred_scores)

        logger.info(f"Correlation:\tPearson: {eval_pearson:.4f}\tSpearman: {eval_spearman:.4f}")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_pearson, eval_spearman])

        return eval_spearman
