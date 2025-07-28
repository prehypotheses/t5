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

        self.__parts = self.__get_parts(connector=connector)

    def __get_parts(self, connector: boto3.session.Session):

        self.__experiment_name = 'FEW'
        self.__experiment_tags = {
            'project': 'custom token classification', 'type': 'natural language processing',
            'task': 'token classification',
            'description': 'The fine-tuning of pre-trained large language model architectures for token classification tasks.'}

        secret = src.functions.secret.Secret(connector=connector)

        t_bucket = secret.exc(secret_id='FNTC', node='tracking-bucket')
        t_secret  = secret.exc(secret_id='FNTC', node='tracking-secret')
        t_endpoint = secret.exc(secret_id='FNTC', node='tracking-endpoint')
        t_database = secret.exc(secret_id='FNTC', node='tracking-database')

        secret.exc(secret_id=t_secret, node='username')
        secret.exc(secret_id=t_secret, node='password')

        return ''


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
