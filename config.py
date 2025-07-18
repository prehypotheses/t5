"""config.py"""
import datetime
import logging
import os
import time


class Config:
    """
    Config
    """

    def __init__(self) -> None:
        """
        Constructor<br>
        -----------<br>

        Variables denoting a path - including or excluding a filename - have an underscore suffix; this suffix is
        excluded for names such as warehouse, storage, depository, etc.<br><br>
        """

        # Directories
        self.data_ = os.path.join(os.getcwd(), 'data')
        self.warehouse = os.path.join(os.getcwd(), 'warehouse')
        self.artefacts_ = os.path.join(self.warehouse, 'artefacts')

        # Prefixes
        self.destination = f'artefacts/T5'

        # Keys, etc
        self.s3_parameters_key = 's3_parameters.yaml'
        self.arguments_key = 'architecture/t5/arguments.json'
        self.hyperspace_key = 'architecture/t5/hyperspace.json'

        '''
       The metadata of the modelling artefacts
       '''
        self.metadata = {'description': 'The modelling artefacts of {architecture}.',
                         'details': 'The {architecture} collection consists of (a) the checkpoints, (b) the logs ' +
                                    'for TensorBoard examination, and (c) much more.'}
