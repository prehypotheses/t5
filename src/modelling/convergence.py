"""Module convergence.py"""
import typing

import transformers

import src.elements.arguments as ag
import src.elements.master as mr
import src.modelling.args
import src.modelling.metrics
import src.modelling.tokenizer


class Convergence:
    """
    Interface
    """

    def __init__(self, arguments: ag.Arguments, master: mr.Master):
        """

        :param arguments:
        :param master:
        """

        self.__arguments = arguments

        # For the tags & data (datasets.DatasetDict)
        self.__data = master.data
        self.__id2label = master.id2label
        self.__label2id = master.label2id

    # pylint: disable=R0801
    def __model(self):
        """

        :return:
        """

        config = transformers.AutoConfig.from_pretrained(
            self.__arguments.pretrained_model_name,
            **{'num_labels': len(self.__id2label), 'label2id': self.__label2id, 'id2label': self.__id2label})

        return transformers.T5ForTokenClassification.from_pretrained(
            self.__arguments.pretrained_model_name, config=config)

    # noinspection DuplicatedCode
    # pylint: disable=R0801
    def __call__(self, branch: typing.Literal['hyperparameters', 'optimal']):
        """

        :param branch: Per model development experiment, artefacts are stored within the experiment's
                       `hyperparameters` or `optimal` directory branch.
        :return:
        """

        metrics = src.modelling.metrics.Metrics(id2label=self.__id2label)
        tokenizer = src.modelling.tokenizer.Tokenizer(arguments=self.__arguments).__call__()

        # Training Arguments
        args = src.modelling.args.Args(
            arguments=self.__arguments, n_instances=self.__data['train'].num_rows).__call__(branch=branch)

        # Data Collator
        data_collator: transformers.DataCollatorForTokenClassification = (
            transformers.DataCollatorForTokenClassification(tokenizer=tokenizer))

        # The training object
        trainer = transformers.trainer.Trainer(
            model=self.__model(), args=args, data_collator=data_collator,
            train_dataset=self.__data['train'], eval_dataset=self.__data['validation'],
            compute_metrics=metrics.exc, callbacks=[transformers.EarlyStoppingCallback(
                early_stopping_patience=self.__arguments.early_stopping_patience)])

        trainer.train()

        return trainer
