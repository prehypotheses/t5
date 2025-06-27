
import logging

import src.data.interface
import src.elements.master as mr
import src.elements.s3_parameters as s3p


class Interface:

    def __init__(self, s3_parameters: s3p.S3Parameters):
        """

        :param s3_parameters:
        """

        self.__master: mr.Master = src.data.interface.Interface(
            s3_parameters=s3_parameters).exc()

    def exc(self):
        """

        :return:
        """

        logging.info(self.__master.id2label)
        logging.info(self.__master.label2id)
        logging.info(self.__master.data)
        logging.info(self.__master.data['train'].features)
