
import ray.tune

class Check:

    def __init__(self):

        self.__num_to_keep = 5
        self.__checkpoint_score_attribute = 'training_iteration'

    def __call__(self):
        """
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CheckpointConfig.html

        :return:
        """

        ray.tune.CheckpointConfig(num_to_keep=self.__num_to_keep,
                                  checkpoint_score_attribute=self.__checkpoint_score_attribute)