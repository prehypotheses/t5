"""Module measurements.py"""
import logging
import os

import sklearn.metrics as sm

import src.functions.directories
import src.functions.objects
import src.valuate.numerics


class Measurements:
    """
    For classification metrics calculations
    """

    def __init__(self, originals: list[str], predictions: list[str]):
        """

        :param originals: The true values, a simple, i.e., un-nested, list.<br>
        :param predictions: The predictions, a simple list, i.e., un-nested, list.<br>
        """

        self.__originals = originals
        self.__predictions = predictions

        # Instances
        self.__objects = src.functions.objects.Objects()

    def __sci(self, path: str):
        """

        :param path: Storage path
        :return:
        """

        report = sm.classification_report(y_true=self.__originals, y_pred=self.__predictions, zero_division=0.0)
        with open(file=os.path.join(path, 'fine.txt'), mode='w', encoding='utf-8') as disk:
            disk.write(report)

        # Preview
        logging.info('scikit-learn:\n%s', report)

    def __numerics(self, path: str) -> None:
        """

        :param path: Storage path
        :return:
        """

        values: dict = src.valuate.numerics.Numerics(
            originals=self.__originals, predictions=self.__predictions).exc()
        self.__objects.write(nodes=values, path=os.path.join(path, 'fundamental.json'))

        # Preview
        logging.info('numerics:\n%s', values)

    def exc(self, path: str):
        """

        :param path: path segment
        :return:
        """

        src.functions.directories.Directories().create(path=path)

        self.__sci(path=path)
        self.__numerics(path=path)
