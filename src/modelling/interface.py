"""Module interface.py"""
import logging

import ray
import transformers

import src.data.interface
import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.s3_parameters as s3p
import src.modelling.args
import src.modelling.check
import src.modelling.metrics
import src.modelling.tuning
import src.modelling.tokenizer


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
        checkpoint_config = src.modelling.check.Check().__call__()
        tokenizer = src.modelling.tokenizer.Tokenizer(arguments=self.__arguments).__call__()

        # Data
        data = self.__bytes.data()
        train_dataset = ray.data.from_huggingface(data['train'])
        eval_dataset = ray.data.from_huggingface(data['validation'])

        # Training Arguments
        args = src.modelling.args.Args(arguments=self.__arguments, n_instances=data['train'].num_rows).__call__()

        # Data Collator
        data_collator: transformers.DataCollatorForTokenClassification = (
            transformers.DataCollatorForTokenClassification(tokenizer=tokenizer))

        # The training object
        trainer = transformers.trainer.Trainer(
            model_init=self.__model_init, args=args, data_collator=data_collator,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            compute_metrics=metrics.exc, callbacks=[transformers.EarlyStoppingCallback(
                early_stopping_patience=self.__arguments.early_stopping_patience)])

        # https://docs.ray.io/en/latest/train/getting-started-transformers.html#report-checkpoints-and-metrics
        # trainer.add_callback(rtht.RayTrainReportCallback())
        # trainer = rtht.prepare_trainer(trainer=trainer)

        # The tuning objects for model training/development
        tuning = src.modelling.tuning.Tuning(arguments=self.__arguments, hyperspace=self.__hyperspace)

        # Hence, hyperparameter search via ...
        best = trainer.hyperparameter_search(
            hp_space=tuning.hp_space, compute_objective=tuning.compute_objective,
            n_trials=self.__arguments.N_TRIALS, direction='minimize', backend='ray',
            name='hyperparameters', resources_per_trial={'cpu': self.__arguments.N_CPU, 'gpu': self.__arguments.N_GPU},
            storage_path=self.__arguments.storage_path, scheduler=tuning.scheduler(), reuse_actors=True,
            checkpoint_config=checkpoint_config,
            verbose=0, progress_reporter=tuning.reporting, log_to_file=True)

        return best
