import glob
import os

import config
import src.elements.arguments as ag
import src.functions.directories


class Structure:
    """
    Structure
    """

    def __init__(self, arguments: ag.Arguments):
        """

        :param arguments:
        """

        self.__arguments = arguments

        # Configurations
        self.__configurations = config.Config()

        # Instances
        self.__directories = src.functions.directories.Directories()

    @staticmethod
    def __name(pathstr: str):
        """

        :param pathstr:
        :return:
        """

        left = pathstr.split('_learning_rate', 1)
        name = left[0]

        return name

    def __stores(self):
        """
        Deletes the runs & checkpoints directories of the hyperparameter search stage.

        :return:
        """

        # Runs
        runs_: str = os.path.join(self.__arguments.model_output_directory, 'hyperparameters', 'run*')
        runs = glob.glob(pathname=runs_, recursive=True)

        # Checkpoints
        checkpoints_: str = os.path.join(
            self.__arguments.model_output_directory, 'hyperparameters', 'compute', '**', 'checkpoint_*')
        checkpoints = glob.glob(pathname=checkpoints_, recursive=True)

        # Hence, altogether
        directories = runs + checkpoints

        # Delete
        for directory in directories:
            self.__directories.cleanup(directory)

    def __renaming(self):
        """
        Renames the objective directories because their default names are too long.

        :return:
        """

        # The directories that start with _objective; add a directory check
        elements = glob.glob(pathname=os.path.join(self.__configurations.artefacts_, '**', '_objective*'), recursive=True)
        directories = [element for element in elements if os.path.isdir(element)]

        # Bases
        bases = [os.path.basename(directory) for directory in directories]
        bases = [self.__name(base) for base in bases]

        # Endpoints
        endpoints = [os.path.dirname(directory) for directory in directories]

        # Rename
        for directory, base, endpoint in zip(directories, bases, endpoints):
            os.rename(src=directory, dst=os.path.join(endpoint, base))

    def exc(self) -> None:
        """

        :return:
        """

        self.__stores()
        self.__renaming()
