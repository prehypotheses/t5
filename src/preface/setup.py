"""Module setup.py"""
import config
import src.functions.cache
import src.functions.directories


class Setup:
    """
    Description
    -----------

    Sets up local & cloud environments
    """

    def __init__(self):
        """

        Constructor
        """

        # Configurations, etc.
        self.__configurations = config.Config()

    def __local(self) -> bool:
        """

        :return:
        """

        directories = src.functions.directories.Directories()
        directories.cleanup(self.__configurations.warehouse)

        return directories.create(self.__configurations.artefacts_)

    def exc(self) -> bool:
        """

        :return:
        """

        return self.__local()
