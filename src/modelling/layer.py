
import logging

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
