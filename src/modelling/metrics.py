"""Module metrics"""

import typing

import transformers

import numpy as np
import pandas as pd
import sklearn

import src.modelling.derivations


class Metrics:
    """

    https://huggingface.co/spaces/evaluate-metric/seqeval/blob/main/seqeval.py
    """

    def __init__(self, id2label: dict):
        """

        :param id2label:
        """

        self.__id2label = id2label
        self.__labels = list(id2label.values())
        self.__fields = ['label', 'N', 'precision', 'sensitivity', 'fnr', 'f-score', 'matthews', 'b-accuracy']

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

    def __cases(self, _predictions, _labels):
        """

        :param _predictions:
        :param _labels:
        :return:
        """

        predictions_ = sum(_predictions, [])
        labels_ = sum(_labels, [])
        matrix = sklearn.metrics.confusion_matrix(y_true=labels_, y_pred=predictions_, labels=self.__labels)

        tp = np.diag(matrix, k=0)
        tn = [int(matrix.sum() - matrix[:,k].sum() - matrix[k,:].sum() + matrix[k,k])
              for k in range(matrix.shape[0])]
        fp = np.sum(matrix, axis=0) - np.diag(matrix, k=0)
        fn = np.sum(matrix, axis=1) - np.diag(matrix, k=0)

        frame = pd.DataFrame(
            data={'label': self.__labels, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'N': np.sum(matrix, axis=1)})

        return frame

    def __publish(self, derivations: pd.DataFrame) -> dict:
        """

        :param derivations: A data frame of error measures & metrics
        :return:
        """

        focus = derivations.copy().loc[derivations['label'].isin(self.__labels), self.__fields]

        publish = focus.melt(id_vars='label', var_name='metric', value_name='score')
        publish = publish.assign(valuation = publish['label'] + '-' + publish['metric'])
        publish.sort_values(by='valuation', ascending=True, inplace=True)

        dictionary = publish[['valuation', 'score']].set_index(
            keys='valuation').to_dict(orient='dict')['score']

        return dictionary

    def exc(self, bucket: transformers.trainer_utils.EvalPrediction) -> dict | None:
        """

        :param bucket: An epoch's prediction output
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
        cases = self.__cases(_predictions=_predictions, _labels=_labels)
        derivations = src.modelling.derivations.Derivations(cases=cases).exc()
        publish = self.__publish(derivations=derivations)

        return publish
