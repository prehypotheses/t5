"""Module persist.py"""
import logging
import os

import datasets

import config
import src.elements.arguments as ag
import src.elements.s3_parameters as s3p


class Persist:
    """
    Persist
    """

    def __init__(self, s3_parameters: s3p.S3Parameters, arguments: ag.Arguments):
        """
        
        :param s3_parameters: The overarching S3 parameters settings of this
                              project, e.g., region code name, buckets, etc.
        :param arguments: Refer to src.elements.arguments
        """

        self.__s3_parameters: s3p.S3Parameters = s3_parameters
        self.__arguments = arguments

        # Configurations
        self.__configurations = config.Config()

    def exc(self, data: datasets.DatasetDict) -> None:
        """

        :param data: The data for model development & evaluation
        :return:
        """

        # The model output directory includes the [temporary] local storage area, which is
        # encoded by self.__configurations.warehouse; this statement removes this local path
        difference = self.__arguments.model_output_directory.replace(self.__configurations.warehouse, '')
        difference = difference.replace(os.sep, '/')

        # Hence, construct the simple storage service string
        dataset_dict_path = 's3://' + self.__s3_parameters.internal + difference + '/data'
        data.save_to_disk(dataset_dict_path=dataset_dict_path)

        logging.info('The data tokens for T5 have been written to prefix: %s', difference + '/data')
