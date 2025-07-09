"""Module interface.py"""
import typing

import datasets

import config
import src.elements.s3_parameters as s3p

import src.functions.directories


class Interface:
    """
    Reads the raw data.
    """

    def __init__(self, s3_parameters: s3p, persist: bool = False):
        """

        :param s3_parameters: The overarching S3 parameters settings of this
                              project, e.g., region code name, buckets, etc.<br>
        :param persist: Save a copy of the downloaded data?
        """

        self.__s3_parameters: s3p.S3Parameters = s3_parameters
        self.__persist = persist

        # Configurations
        self.__configurations = config.Config()

        # The data
        dataset_path = 's3://' + self.__s3_parameters.internal + '/' + self.__configurations.source
        self.__data =  datasets.load_from_disk(dataset_path=dataset_path)

    def tags(self) -> typing.Tuple[dict, dict]:
        """

        :return:<br>
            id2label: dict<br>
            label2id: dict
        """

        values: datasets.Sequence = self.__data['train'].features['fine_ner_tags']
        id2label = dict(enumerate(values.feature.names))
        label2id = {value: key for key, value in id2label.items()}

        return id2label, label2id

    def data(self) -> datasets.DatasetDict:
        """

        :return:
        """

        # Persist
        if self.__persist:
            directories = src.functions.directories.Directories()
            directories.cleanup(self.__configurations.tokens_)
            directories.create(self.__configurations.tokens_)
            self.__data.save_to_disk(self.__configurations.tokens_)

        return self.__data
