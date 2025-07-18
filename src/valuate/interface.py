"""Module interface.py"""
import datasets
import transformers

import src.valuate.estimates
import src.valuate.measurements


class Interface:
    """
    Interface
    """

    def __init__(self, model: transformers.Trainer, id2label: dict):
        """

        :param model:
        :param id2label:
        """

        self.__model = model
        self.__id2label = id2label

    def exc(self, blob: datasets.Dataset, path: str):
        """

        :param blob:
        :param path:
        :return:
        """

        originals, predictions = src.valuate.estimates.Estimates(
            blob=blob, id2label=self.__id2label).exc(model=self.__model)

        src.valuate.measurements.Measurements(
            originals=originals, predictions=predictions).exc(path=path)
