"""Module interface.py"""
import os

import datasets
import transformers

import src.elements.arguments as ag
import src.functions.secret
import src.valuate.estimates
import src.valuate.measurements


class Interface:
    """
    Interface
    """

    def __init__(self, model: transformers.Trainer, id2label: dict, arguments: ag.Arguments):
        """

        :param model:
        :param id2label:
        :param arguments:
        """

        self.__model = model
        self.__id2label = id2label
        self.__arguments = arguments

        self.__experiment_name = 'FEW'
        self.__experiment_tags = {
            'project': 'custom token classification', 'type': 'natural language processing',
            'task': 'token classification', 'team': 'data science core',
            'description': 'Token classification via the fine tuning of pre-trained large language model architectures'}

    def __parts(self):

        pass


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
