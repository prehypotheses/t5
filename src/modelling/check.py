"""Module check.py"""
import ray.tune


class Check:
    """
    Creates a checkpoints configuration
    """

    def __init__(self):
        """
        Constructor
        """

        self.__num_to_keep = 5
        self.__checkpoint_score_attribute = 'training_iteration'

    def __call__(self) -> ray.tune.CheckpointConfig:
        """
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CheckpointConfig.html

        :return:
        """

        return ray.tune.CheckpointConfig(
            num_to_keep=self.__num_to_keep,
            checkpoint_score_attribute=self.__checkpoint_score_attribute)
