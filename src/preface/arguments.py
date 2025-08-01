"""Module arguments.py"""
import datetime
import os
import time

import boto3

import config
import src.elements.arguments as ag
import src.s3.configurations


class Arguments:
    """
    Returns a suite of values/arguments for machine learning model development.<br>
    """

    def __init__(self):

        self.__configurations = config.Config()

        # Stamp
        today: datetime.datetime = datetime.datetime.now()
        self.__seconds = int(time.mktime(today.timetuple()))

    def __get_arguments(self, connector: boto3.session.Session) -> ag.Arguments:
        """

        :param connector: A boto3 session instance, it retrieves the developer's <default> Amazon
                          Web Services (AWS) profile details, which allows for programmatic interaction with AWS.
        :return:
        """

        dictionary = src.s3.configurations.Configurations(connector=connector).objects(
            key_name=self.__configurations.arguments_key)

        # Set up the model output directory parameter
        dictionary['experiment_segment'] = str(self.__seconds)
        dictionary['model_output_directory'] = os.path.join(
            self.__configurations.artefacts_, dictionary['architecture'].upper(), dictionary['experiment_segment'])

        return ag.Arguments(**dictionary)

    def __call__(self, connector: boto3.session.Session):
        """

        :param connector:
        :return:
        """

        return self.__get_arguments(connector=connector)
