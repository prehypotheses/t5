"""Module interface.py"""
import os
import typing
import datetime
import time

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
import src.preface.hyperspace


class Interface:
    """
    Interface
    """

    def __init__(self):
        """
        Constructor
        """

        self.__configurations = config.Config()

        today = datetime.datetime.now().strftime('%Y-%m-%d')
        pattern = datetime.datetime.strptime(today, '%Y-%m-%d %H:%M:%S')
        self.__seconds = int(time.mktime(pattern.timetuple()))

        self.__hyperspace = src.preface.hyperspace.Hyperspace()

    def __arguments(self, connector: boto3.session.Session) -> ag.Arguments:
        """

        :param connector:
        :return:
        """

        dictionary = src.s3.configurations.Configurations(connector=connector).objects(
            key_name=self.__configurations.arguments_key)

        # Set up the model output directory parameter
        dictionary['experiment_segment'] = str(self.__seconds)
        dictionary['model_output_directory'] = os.path.join(
            self.__configurations.artefacts_, dictionary['architecture'].upper(), dictionary['experiment_segment'])

        return ag.Arguments(**dictionary)

    def exc(self) -> typing.Tuple[boto3.session.Session, s3p.S3Parameters, sr.Service, ag.Arguments, hp.Hyperspace]:
        """

        :return:
        """

        # Cloud services instances
        connector = boto3.session.Session()
        s3_parameters: s3p.S3Parameters = src.s3.s3_parameters.S3Parameters(connector=connector).exc()
        service: sr.Service = src.functions.service.Service(
            connector=connector, region_name=s3_parameters.region_name).exc()

        # Arguments
        arguments = self.__arguments(connector=connector)

        # Setting up the cloud storage area
        prefix = arguments.model_output_directory.replace(self.__configurations.warehouse, '')
        prefix = prefix.replace(os.sep, '/')
        src.preface.setup.Setup(service=service, s3_parameters=s3_parameters, prefix=prefix).exc()

        return (connector, s3_parameters, service, arguments,
                self.__hyperspace(connector=connector))
