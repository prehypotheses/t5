import transformers
import ray

import src.modelling.tokenizer

import src.data.interface
import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.s3_parameters as s3p


class Intelligence:

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
        self.__id2label, self.__label2id = self.__bytes.tags()

    def __model(self):
        """

        :return:
        """

        config = transformers.AutoConfig.from_pretrained(
            self.__arguments.pretrained_model_name,
            **{'num_labels': len(self.__id2label), 'label2id': self.__label2id, 'id2label': self.__id2label,
               'dense_act_fn': 'gelu'})

        return transformers.T5ForTokenClassification.from_pretrained(
            self.__arguments.pretrained_model_name, config=config)

    def train_func(self):

        tokenizer = src.modelling.tokenizer.Tokenizer(arguments=self.__arguments).__call__()
        model = self.__model()

        # Data
        data = self.__bytes.data()
        train_dataset = ray.data.from_huggingface(data['train'])
        eval_dataset = ray.data.from_huggingface(data['validation'])
