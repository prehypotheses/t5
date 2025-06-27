"""Module interface.py"""
import os
import typing

import boto3

import config
import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.functions.service
import src.preface.setup
import src.s3.configurations
import src.s3.s3_parameters


class Interface:
    """
    Interface
    """

    def __init__(self):
        """
        Constructor
        """

        self.__configurations = config.Config()

    def __arguments(self, connector: boto3.session.Session) -> ag.Arguments:
        """

        :param connector:
        :return:
        """

        dictionary = src.s3.configurations.Configurations(connector=connector).objects(
            key_name=self.__configurations.arguments_key)

        # Set up the model output directory parameter
        model_output_directory = os.path.join(self.__configurations.artefacts_, dictionary['architecture'])
        dictionary['model_output_directory'] = model_output_directory

        return ag.Arguments(**dictionary)

    def __hyperspace(self, connector: boto3.session.Session) -> hp.Hyperspace:
        """

        :param connector:
        :return:
        """

        dictionary = src.s3.configurations.Configurations(connector=connector).objects(
            key_name=self.__configurations.hyperspace_key)

        items = {'learning_rate_distribution': dictionary['continuous']['learning_rate'],
                 'weight_decay_distribution': dictionary['continuous']['weight_decay'],
                 'weight_decay_choice': dictionary['choice']['weight_decay'],
                 'per_device_train_batch_size': dictionary['choice']['per_device_train_batch_size']}

        # Hence
        return hp.Hyperspace(**items)

    def exc(self) -> typing.Tuple[boto3.session.Session, s3p.S3Parameters, sr.Service, ag.Arguments, hp.Hyperspace]:
        """

        :return:
        """

        # Cloud services instances
        connector = boto3.session.Session()
        s3_parameters: s3p.S3Parameters = src.s3.s3_parameters.S3Parameters(connector=connector).exc()
        service: sr.Service = src.functions.service.Service(
            connector=connector, region_name=s3_parameters.region_name).exc()

        # Setting up the cloud storage area
        src.preface.setup.Setup(service=service, s3_parameters=s3_parameters).exc()

        return (connector, s3_parameters, service, self.__arguments(connector=connector),
                self.__hyperspace(connector=connector))
