"""Module convergence.py"""
import logging

import transformers

import src.data.interface
import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.s3_parameters as s3p
import src.modelling.args
import src.modelling.check
import src.modelling.metrics
import src.modelling.tokenizer
import src.modelling.tuning


# noinspection DuplicatedCode
class Convergence:
    """
    Interface
    """

    def __init__(self, s3_parameters: s3p.S3Parameters, arguments: ag.Arguments, hyperspace: hp.Hyperspace,
                 pieces: src.data.interface.Interface):
        """

        :param s3_parameters:
        :param arguments:
        :param hyperspace:
        :param pieces:
        """

        self.__s3_parameters = s3_parameters
        self.__arguments = arguments
        self.__hyperspace = hyperspace

        # For the tags & data (datasets.DatasetDict)
        self.__data = pieces.data()
        self.__id2label, self.__label2id = pieces.tags()

    def __model_init(self):
        """

        :return:
        """

        config = transformers.AutoConfig.from_pretrained(
            self.__arguments.pretrained_model_name,
            **{'num_labels': len(self.__id2label), 'label2id': self.__label2id, 'id2label': self.__id2label,
               'dense_act_fn': 'gelu'})

        return transformers.T5ForTokenClassification.from_pretrained(
            self.__arguments.pretrained_model_name, config=config)

    def __call__(self):
        """

        :return:
        """

        metrics = src.modelling.metrics.Metrics(id2label=self.__id2label)
        tokenizer = src.modelling.tokenizer.Tokenizer(arguments=self.__arguments).__call__()

        # Training Arguments
        args = src.modelling.args.Args(arguments=self.__arguments, n_instances=self.__data['train'].num_rows).__call__()

        # Data Collator
        data_collator: transformers.DataCollatorForTokenClassification = (
            transformers.DataCollatorForTokenClassification(tokenizer=tokenizer))

        # The training object
        trainer = transformers.trainer.Trainer(
            model_init=self.__model_init, args=args, data_collator=data_collator,
            train_dataset=self.__data['train'], eval_dataset=self.__data['validation'],
            compute_metrics=metrics.exc, callbacks=[transformers.EarlyStoppingCallback(
                early_stopping_patience=self.__arguments.early_stopping_patience)])

        trainer.train()

        return trainer
