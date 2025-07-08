
import logging
import transformers

import src.data.interface
import src.modelling.args
import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.s3_parameters as s3p
import ray


class Interface:
    """
    Interface
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

        # For the tags, id2label & label2id, and the datasets.DatasetDict
        self.__bytes = src.data.interface.Interface(s3_parameters=s3_parameters)

    def model_init(self):
        """

        :return:
        """

        id2label, label2id = self.__bytes.tags()

        config = transformers.AutoConfig.from_pretrained(
            self.__arguments.pretrained_model_name,
            **{'num_labels': len(id2label), 'label2id': label2id, 'id2label': id2label,   'dense_act_fn': 'gelu'})

        return transformers.T5ForTokenClassification.from_pretrained(
            self.__arguments.pretrained_model_name, config=config)

    def exc(self):
        """

        :return:
        """

        data = self.__bytes.data()
        train = ray.data.from_huggingface(data['train'])
        validation = ray.data.from_huggingface(data['validation'])

        args = src.modelling.args.Args(arguments=self.__arguments).__call__()
