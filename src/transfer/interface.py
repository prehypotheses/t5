"""Module transfer.py"""
import glob
import os

import config
import src.transfer.dictionary
import src.elements.s3_parameters as s3p
import src.elements.service as sr
import src.functions.directories
import src.s3.ingress


class Interface:
    """
    Transfers data files to an Amazon S3 (Simple Storage Service) prefix.

    """

    def __init__(self, service: sr.Service,  s3_parameters: s3p, architecture: str):
        """

        :param service: A suite of services for interacting with Amazon Web Services.
        :param s3_parameters: The overarching S3 parameters settings of this
                              project, e.g., region code name, buckets, etc.
        :param architecture: The pre-trained model architecture in focus.
        """

        self.__service: sr.Service = service
        self.__s3_parameters: s3p.S3Parameters = s3_parameters
        self.__architecture: str = architecture

        # Instances
        self.__configurations = config.Config()
        self.__dictionary = src.transfer.dictionary.Dictionary(architecture=architecture)
        self.__directories = src.functions.directories.Directories()

    @staticmethod
    def __name(pathstr: str):
        """

        :param pathstr:
        :return:
        """

        left = pathstr.split('_', maxsplit=4)
        right = pathstr.rsplit('_', maxsplit=2)
        strings = left[1:3] + right[-2:]
        name = '_'.join(strings)

        return name

    def __stores(self):
        """
        Deletes the runs & checkpoints directories of the hyperparameter search stage.

        :return:
        """

        # Runs
        runs_: str = os.path.join(self.__configurations.artefacts_, self.__architecture, 'hyperparameters', 'run*')
        runs = glob.glob(pathname=runs_, recursive=True)

        # Checkpoints
        checkpoints_: str = os.path.join(
            self.__configurations.artefacts_, self.__architecture, 'hyperparameters', 'compute', '**', 'checkpoint_*')
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

    def exc(self):
        """

        :return:
        """

        # Foremost (a) delete runs & checkpoints data, and (b) rename the <_objective*> directories.
        self.__stores()
        self.__renaming()

        # The details of the data being transferred to Amazon S3 (Simple Storage Service)
        strings = self.__dictionary.exc(
            path=self.__configurations.artefacts_, extension='*', prefix=self.__s3_parameters.path_internal_artefacts)

        # Transfer
        messages = src.s3.ingress.Ingress(
            service=self.__service, bucket_name=self.__s3_parameters.internal).exc(strings=strings, tagging='project=few')

        return messages
