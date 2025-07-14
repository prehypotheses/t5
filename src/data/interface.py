"""Module interface.py"""
import typing
import warnings

import datasets

import config
import src.elements.arguments as ag
import src.elements.s3_parameters as s3p
import src.elements.master as mr
import src.data.tags


class Interface:
    """
    Reads the raw data.
    """

    def __init__(self, s3_parameters: s3p, arguments: ag.Arguments, persist: bool = True):
        """

        :param s3_parameters: The overarching S3 parameters settings of this
                              project, e.g., region code name, buckets, etc.<br>
        :param arguments: Refer to src.elements.arguments
        :param persist: Save a copy of the downloaded data?
        """

        self.__s3_parameters: s3p.S3Parameters = s3_parameters
        self.__arguments = arguments
        self.__persist = persist

        # Configurations
        self.__configurations = config.Config()

    def __get_data(self) -> datasets.DatasetDict:
        """

        :return:
        """

        # The data
        dataset_path = 's3://' + self.__s3_parameters.internal + '/' + self.__arguments.raw_
        warnings.filterwarnings("ignore", message="promote has been superseded by promote_options='default'.",
                                category=FutureWarning, module="awswrangler")

        return datasets.load_from_disk(dataset_path=dataset_path)

    def __filter(self, data: datasets.DatasetDict) -> datasets.DatasetDict:
        """

        :param data:
        :return:
        """

        excerpt = data.copy()
        for section in ['train', 'validation', 'test']:
            excerpt[section] = excerpt[section].shuffle(seed=self.__arguments.seed).select(
                range(int(self.__arguments.fraction * excerpt[section].num_rows)))
        data = datasets.DatasetDict(excerpt)

        return data

    def __persist_(self, excerpt: datasets.DatasetDict):
        """

        :param excerpt:
        :return:
        """

        excerpt.save_to_disk(self.__configurations.data_)

    def exc(self) -> mr.Master:
        """

        :return:
        """

        # A datasets.DatasetDict consisting of `train`, `validation`, & `test` datasets.Dataset objects.
        data = self.__get_data()
        excerpt = self.__filter(data=data) if self.__arguments.fraction < 1 else data

        # Persist
        if self.__persist:
            self.__persist_(excerpt=excerpt)

        # Tags
        id2label, label2id = src.data.tags.Tags().exc(feed=excerpt['train'])

        return mr.Master(id2label=id2label, label2id=label2id, data=excerpt)
