"""config.py"""
import os


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

        # Keys, etc
        self.s3_parameters_key = 's3_parameters.yaml'

        # Prefixes
        self.source = 'data/tokens'
        self.destination = 'artefacts/t5'

        # For arguments JSON
        self.checkpoint = 'google-t5/t5-small'
