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
        self.artefacts_ = os.path.join(self.warehouse, 'artefacts')

        # Keys, etc
        self.s3_parameters_key = 's3_parameters.yaml'
        self.arguments_key = 'architecture/t5/arguments.json'
        self.hyperspace_key = 'architecture/t5/hyperspace.json'

        # Prefixes
        self.source = 'data/tokens/T5'
        self.destination = 'artefacts/T5'
