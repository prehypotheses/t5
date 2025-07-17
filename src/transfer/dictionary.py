"""Module dictionary.py"""
import glob
import os

import numpy as np
import pandas as pd

import config


class Dictionary:
    """
    Class KeyStrings
    """

    def __init__(self, architecture: str):
        """

        :param architecture:
        """

        self.__metadata = config.Config().metadata
        self.__metadata = {k: v.format(architecture=architecture)
                           for k, v in self.__metadata.items()}

    @staticmethod
    def __local(path: str, extension: str) -> pd.DataFrame:
        """

        :param path: The path wherein the files of interest lie
        :param extension: The extension type of the files of interest
        :return:
        """

        splitter = os.path.basename(path) + os.path.sep

        # The list of files within the path directory, including its child directories.
        files: list[str] = glob.glob(pathname=os.path.join(path, '**',  f'*.{extension}'),
                                     recursive=True)

        details: list[dict] = [
            {'file': file,
             'vertex': file.rsplit(splitter, maxsplit=1)[1]}
            for file in files]

        return pd.DataFrame.from_records(details)

    def exc(self, path: str, extension: str, prefix: str) -> pd.DataFrame:
        """

        :param path: The path wherein the files of interest lie
        :param extension: The extension type of the files of interest
        :param prefix: The Amazon S3 (Simple Storage Service) where the files of path are heading
        :return:
        """

        local: pd.DataFrame = self.__local(path=path, extension=extension)

        # Building the Amazon S3 strings
        frame = local.assign(key=prefix + local["vertex"])

        # The metadata dict strings
        frame['metadata'] = np.array(self.__metadata).repeat(frame.shape[0])

        return frame[['file', 'key', 'metadata']]
