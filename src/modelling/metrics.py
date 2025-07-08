
import collections

import transformers
import numpy as np
import evaluate
import typing


class Metrics:
    """

    https://huggingface.co/spaces/evaluate-metric/seqeval/blob/main/seqeval.py
    """

    def __init__(self, id2label: dict):
        """

        :param id2label:
        """

        self.__id2label = id2label
        self.__seqeval = evaluate.load('seqeval')

    def __active(self, predictions: np.ndarray, labels: np.ndarray) -> typing.Tuple[list[list], list[list]]:
        """

        :param predictions:
        :param labels:
        :return:
        """

        _predictions = [
            [self.__id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        _labels = [
            [self.__id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return _predictions, _labels

    @staticmethod
    def __restructure(key: str, dictionary: dict):
        """

        :param key:
        :param dictionary:
        :return:
        """

        return {f'{key}_{k}': v for k, v in dictionary.items()}

    def __decompose(self, metrics: dict) -> dict:
        """

        :param metrics:{<br>
                    &nbsp; 'class<sub>1</sub>': {'metric<sub>1</sub>': value, 'metric<sub>2</sub>': value, ...},<br>
                    &nbsp; 'class<sub>2</sub>': {'metric<sub>1</sub>': value, 'metric<sub>2</sub>': value, ...}, ...}
        :return:
        """

        # Class level metrics
        disaggregates = {k: v for k, v in metrics.items() if not k.startswith('overall')}

        # Re-structuring the dictionary of class level metrics
        metrics_per_class = list(map(lambda x: self.__restructure(x[0], x[1]), disaggregates.items()))

        # Overarching metrics
        aggregates = {k: v for k, v in metrics.items() if k.startswith('overall')}

        return dict(collections.ChainMap(*metrics_per_class, aggregates))

    def exc(self, bucket: transformers.trainer_utils.PredictionOutput):
        """

        :param bucket:
        :return:
        """

        # Predictions
        predictions = bucket.predictions
        predictions = np.argmax(predictions, axis=2)

        # Labels
        labels = bucket.label_ids

        # Active
        _predictions, _labels = self.__active(predictions=predictions, labels=labels)

        # Hence
        metrics = self.__seqeval.compute(predictions=_predictions, references=_labels, zero_division=0.0)
        decomposition = self.__decompose(metrics=metrics)


        return decomposition
