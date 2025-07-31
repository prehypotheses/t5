"""Module interface.py"""
import os

import datasets
import transformers

import src.elements.arguments as ag
import src.valuate.estimates
import src.valuate.measurements
import src.valuate.lineage


class Interface:
    """
    Interface
    """

    def __init__(self, model: transformers.Trainer, id2label: dict, arguments: ag.Arguments, experiment: dict):
        """

        :param model:
        :param id2label:
        :param arguments:
        :param experiment:
        """

        self.__model = model
        self.__id2label = id2label
        self.__arguments = arguments
        self.__experiment = experiment

    def exc(self, blob: datasets.Dataset, branch: str, stage: str):
        """

        :param blob:
        :param branch:
        :param stage:
        :return:
        """

        path = os.path.join(self.__arguments.model_output_directory, branch, 'metrics', stage)

        originals, predictions = src.valuate.estimates.Estimates(
            blob=blob, id2label=self.__id2label).exc(model=self.__model)

        src.valuate.measurements.Measurements(
            originals=originals, predictions=predictions).exc(path=path)

        src.valuate.lineage.Lineage(id2label=self.__id2label, experiment=self.__experiment).exc(
            originals=originals, predictions=predictions, stage=stage)
