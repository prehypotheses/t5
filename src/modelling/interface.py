
import logging

import src.data.interface
import src.elements.master as mr
import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.s3_parameters as s3p
import ray


class Interface:

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

        master: mr.Master = src.data.interface.Interface(
            s3_parameters=self.__s3_parameters).exc()

        train = ray.data.from_huggingface(master.data['train'])
        validation = ray.data.from_huggingface(master.data['validation'])



