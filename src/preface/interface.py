"""Module interface.py"""

import typing

import boto3

import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.functions.cache
import src.functions.service
import src.preface.setup
import src.s3.configurations
import src.s3.s3_parameters


class Interface:
    """
    Interface
    """

    def __init__(self):
        pass

    @staticmethod
    def exc() -> typing.Tuple[boto3.session.Session, s3p.S3Parameters, sr.Service]:
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

        return connector, s3_parameters, service
