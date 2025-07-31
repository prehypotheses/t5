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
import src.preface.arguments
import src.preface.experiment
import src.preface.hyperspace
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

        # Instances
        self.__arguments = src.preface.arguments.Arguments()
        self.__hyperspace = src.preface.hyperspace.Hyperspace()

    def exc(self) -> typing.Tuple[boto3.session.Session, s3p.S3Parameters, sr.Service, ag.Arguments, hp.Hyperspace, dict]:
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

        # Experiment
        experiment = src.preface.experiment.Experiment(
            connector=connector, arguments=arguments).exc()

        return (connector, s3_parameters, service, arguments,
                self.__hyperspace(connector=connector), experiment)
