
import logging
import os

import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.s3_parameters as s3p

import src.modelling.structures

# noinspection DuplicatedCode
class Layer:
    """
    Layer
    """

    def __init__(self, s3_parameters: s3p.S3Parameters, arguments: ag.Arguments, hyperspace: hp.Hyperspace):
        """

        :param s3_parameters:
        :param arguments:
        :param hyperspace:
        """

        self.__s3_parameters = s3_parameters
        self.__arguments = arguments
        self.__hyperspace = hyperspace

        # Storage Section
        self.__section = self.__arguments.model_output_directory

    def exc(self):
        """

        :return:
        """

        best = src.modelling.structures.Structures(
            s3_parameters=self.__s3_parameters, arguments=self.__arguments, hyperspace=self.__hyperspace).train_func()

        logging.info(best)
        logging.info(best.hyperparameters)
        logging.info(best.run_summary)
        logging.info(best.__dir__())

        # Hence, update the modelling variables
        self.__arguments = self.__arguments._replace(
            LEARNING_RATE=best.hyperparameters.get('learning_rate'),
            WEIGHT_DECAY=best.hyperparameters.get('weight_decay'),
            TRAIN_BATCH_SIZE=best.hyperparameters.get('per_device_train_batch_size'))

        # Additionally, prepare the artefacts storage area for the best model, vis-à-vis best hyperparameters
        # set, and save a checkpoint at the optimal training point only by setting save_total_limit = 1.
        self.__arguments = self.__arguments._replace(
            model_output_directory=os.path.join(self.__section, 'prime'),
            EPOCHS=2*self.__arguments.EPOCHS, save_total_limit=1)
