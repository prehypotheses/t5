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
        :param id2label: A dictionary wherein (a) the keys are the identification codes of text labels,
                         and (b) the values are the labels.
        :param arguments: A suite of values/arguments for machine learning model development.<br>
        :param experiment:
        """

        self.__model = model
        self.__id2label = id2label
        self.__arguments = arguments
        self.__experiment = experiment

        # Lineage
        self.__lineage = src.valuate.lineage.Lineage(id2label=self.__id2label, experiment=self.__experiment)

    def exc(self, blob: datasets.Dataset, branch: str, stage: str):
        """

        :param blob: The data
        :param branch: hyperparameters or optimal
        :param stage: train, validation, or test
        :return:
        """

        path = os.path.join(self.__arguments.model_output_directory, branch, 'metrics', stage)

        originals, predictions = src.valuate.estimates.Estimates(
            blob=blob, id2label=self.__id2label).exc(model=self.__model)

        src.valuate.measurements.Measurements(
            originals=originals, predictions=predictions).exc(path=path)

        self.__lineage.exc(originals=originals, predictions=predictions, stage=stage)
