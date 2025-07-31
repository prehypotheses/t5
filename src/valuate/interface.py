"""Module interface.py"""
import os

import datasets
import transformers

import src.elements.arguments as ag
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

    def exc(self, blob: datasets.Dataset, branch: str, stage: str):
        """
        https://mlflow.org/docs/latest/ml/tracking/backend-stores/#supported-store-types
        https://mlflow.org/docs/latest/ml/getting-started/logging-first-model/step6-logging-a-run/#using-mlflow-tracking-to-keep-track-of-training
        mlflow.set_tracking_uri()

        t_bucket = self.__secret.exc(secret_id='FNTC', node='tracking-bucket')

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
