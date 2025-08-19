"""Module transfer.py"""
import glob
import os

import datasets

import config
import src.elements.arguments as ag
import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.functions.directories
import src.s3.ingress
import src.transfer.cloud
import src.transfer.dictionary
import src.transfer.structure
import src.transfer.persist


class Interface:
    """
    Transfers data files to an Amazon S3 (Simple Storage Service) prefix.
    """

    def __init__(self, service: sr.Service,  s3_parameters: s3p.S3Parameters, arguments: ag.Arguments):
        """

        :param service: A suite of services for interacting with Amazon Web Services.
        :param s3_parameters: The overarching S3 parameters settings of this
                              project, e.g., region code name, buckets, etc.
        :param arguments: Refer to src.elements.arguments
        """

        self.__service: sr.Service = service
        self.__s3_parameters: s3p.S3Parameters = s3_parameters
        self.__arguments: ag.Arguments = arguments

        # Instances
        self.__configurations = config.Config()
        self.__dictionary = src.transfer.dictionary.Dictionary(architecture=self.__arguments.architecture)
        self.__directories = src.functions.directories.Directories()

    def exc(self, data: datasets.DatasetDict):
        """

        :param data:
        :return:
        """

        # Foremost (a) delete runs & checkpoints data, and (b) rename the <_objective*> directories.
        structure = src.transfer.structure.Structure(arguments=self.__arguments)
        structure.exc()

        # The details of the data being transferred to Amazon S3 (Simple Storage Service)
        strings = self.__dictionary.exc(
            path=self.__configurations.artefacts_, extension='*', prefix=self.__s3_parameters.path_internal_artefacts)

        # Setting up the cloud storage area
        src.transfer.cloud.Cloud(
            service=self.__service, s3_parameters=self.__s3_parameters,
            architecture=self.__arguments.architecture.upper()).exc()

        # Transfer
        src.transfer.persist.Persist(
            s3_parameters=self.__s3_parameters, arguments=self.__arguments).exc(data=data)
        messages = src.s3.ingress.Ingress(
            service=self.__service, bucket_name=self.__s3_parameters.internal).exc(strings=strings, tagging='project=few')

        return messages
