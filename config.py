"""config.py"""
import os
import datetime
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
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        pattern = datetime.datetime.strptime(f'{today} 00:00:00', '%Y-%m-%d %H:%M:%S')
        seconds = int(time.mktime(pattern.timetuple()))

        self.destination = f'artefacts/T5/{str(seconds)}'

        # Keys, etc
        self.s3_parameters_key = 's3_parameters.yaml'
        self.arguments_key = 'architecture/t5/arguments.json'
        self.hyperspace_key = 'architecture/t5/hyperspace.json'
