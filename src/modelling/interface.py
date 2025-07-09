"""Module interface.py"""
import logging

import ray
import transformers

import src.data.interface
import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.s3_parameters as s3p
import src.modelling.args
import src.modelling.metrics
import src.modelling.tuning


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
        self.__id2label, self.__label2id = self.__bytes.tags()

        # Metrics
        self.__metrics = src.modelling.metrics.Metrics(id2label=self.__id2label)

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

    def exc(self):
        """

        :return:
        """

        data = self.__bytes.data()
        train_dataset = ray.data.from_huggingface(data['train'])
        eval_dataset = ray.data.from_huggingface(data['validation'])

        args = src.modelling.args.Args(arguments=self.__arguments).__call__()

        trainer = transformers.trainer.Trainer(
            model_init=self.__model_init, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
            compute_metrics=self.__metrics.exc, callbacks=[transformers.EarlyStoppingCallback(
                early_stopping_patience=self.__arguments.early_stopping_patience)])

        tuning = src.modelling.tuning.Tuning(arguments=self.__arguments, hyperspace=self.__hyperspace)

        trainer.hyperparameter_search(
            hp_space=lambda _: tuning.hp_space(), compute_objective=tuning.compute_objective,
            n_trials=self.__arguments.N_TRIALS, direction='minimize', backend='ray',
            resources_per_trial={'cpu': self.__arguments.N_CPU, 'gpu': self.__arguments.N_GPU},
            scheduler=tuning.scheduler())




