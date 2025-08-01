"""Module hyperspace.py"""
import boto3

import config
import src.elements.hyperspace as hp
import src.s3.configurations


class Hyperspace:
    """
    The search space definitions per hyperparameter.
    """

    def __init__(self):
        """
        Constructor
        """

        self.__configurations = config.Config()

    def __get_hyperspace(self, connector: boto3.session.Session) -> hp.Hyperspace:
        """

        :param connector: A boto3 session instance, it retrieves the developer's <default> Amazon
                          Web Services (AWS) profile details, which allows for programmatic interaction with AWS.
        :return:
        """

        dictionary = src.s3.configurations.Configurations(connector=connector).objects(
            key_name=self.__configurations.hyperspace_key)

        items = {'learning_rate_distribution': dictionary['continuous']['learning_rate'],
                 'weight_decay_distribution': dictionary['continuous']['weight_decay'],
                 'weight_decay_choice': dictionary['choice']['weight_decay'],
                 'per_device_train_batch_size': dictionary['choice']['per_device_train_batch_size']}

        # Hence
        return hp.Hyperspace(**items)

    def __call__(self, connector: boto3.session.Session):
        """

        :param connector:
        :return:
        """

        return self.__get_hyperspace(connector=connector)
