"""Module tokenization.py"""
import logging

import src.elements.arguments as ag
import src.elements.master as mr
import src.modelling.mappings
import src.modelling.tokenizer


class Tokenization:
    """
    Tokenization
    """

    def __init__(self, arguments: ag.Arguments):
        """

        :param arguments:
        """

        self.__arguments = arguments

    def __tokenization(self, master: mr.Master):
        """

        :param master:
        :return:
        """

        tokenizer = src.modelling.tokenizer.Tokenizer(arguments=self.__arguments).__call__()
        mappings = src.modelling.mappings.Mappings(tokenizer=tokenizer, _id2label=master.id2label)

        try:
            packets = master.data.map(mappings.exc, batched=True)
        except RuntimeError as err:
            raise err from err

        master = master._replace(data=packets)
        logging.info(master.data)

        return master

    def exc(self, master: mr.Master) -> mr.Master:
        """

        :param master:
        :return:
        """

        return self.__tokenization(master=master)
