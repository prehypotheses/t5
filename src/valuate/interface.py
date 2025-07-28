"""Module interface.py"""
import os
import boto3

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

    def __init__(self, connector: boto3.session.Session, model: transformers.Trainer, id2label: dict, arguments: ag.Arguments):
        """

        :param connector: A boto3 session instance, it retrieves the developer's <default> Amazon
                          Web Services (AWS) profile details, which allows for programmatic interaction with AWS.
        :param model:
        :param id2label:
        :param arguments:
        """

        self.__model = model
        self.__id2label = id2label
        self.__arguments = arguments

        # Instances
        self.__secret = src.functions.secret.Secret(connector=connector)

        # Tracking Resource
        self.__tracking_uri = self.__get_tracking_uri()

    def __get_tracking_uri(self) -> str:

        t_secret  = self.__secret.exc(secret_id='FNTC', node='tracking-secret')
        t_endpoint = self.__secret.exc(secret_id='FNTC', node='tracking-endpoint')
        t_database = self.__secret.exc(secret_id='FNTC', node='tracking-database')
        t_port = self.__secret.exc(secret_id='FNTC', node='tracking-port')

        username = self.__secret.exc(secret_id=t_secret, node='username')
        password = self.__secret.exc(secret_id=t_secret, node='password')

        uri = f"postgresql://{username}:{password}@{t_endpoint}:{t_port}/{t_database}"

        return uri

    def exc(self, blob: datasets.Dataset, branch: str, stage: str):
        """

        :param blob:
        :param branch:
        :param stage:
        :return:
        """

        '''
        https://mlflow.org/docs/latest/ml/tracking/backend-stores/#supported-store-types
        https://mlflow.org/docs/latest/ml/getting-started/logging-first-model/step6-logging-a-run/#using-mlflow-tracking-to-keep-track-of-training
        mlflow.set_tracking_uri()
        
        t_bucket = self.__secret.exc(secret_id='FNTC', node='tracking-bucket')
        
        '''

        path = os.path.join(self.__arguments.model_output_directory, branch, 'metrics', stage)

        originals, predictions = src.valuate.estimates.Estimates(
            blob=blob, id2label=self.__id2label).exc(model=self.__model)

        src.valuate.measurements.Measurements(
            originals=originals, predictions=predictions).exc(path=path)
