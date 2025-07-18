"""Module architecture.py"""
import os
import transformers

import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.master as mr
import src.modelling.args
import src.modelling.check
import src.modelling.metrics
import src.modelling.tokenizer
import src.modelling.tuning


class Architecture:
    """
    Interface
    """

    def __init__(self, arguments: ag.Arguments, hyperspace: hp.Hyperspace,
                 master:  mr.Master):
        """

        :param arguments:
        :param hyperspace:
        :param master:
        """

        self.__arguments = arguments
        self.__hyperspace = hyperspace

        # For the tags, and the datasets.DatasetDict
        self.__data = master.data
        self.__id2label = master.id2label
        self.__label2id = master.label2id

        # The tuning objects for model training/development
        self.__tuning = src.modelling.tuning.Tuning(arguments=self.__arguments, hyperspace=self.__hyperspace)

    # pylint: disable=R0801
    def __model_init(self):
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
    def train_func(self) -> transformers.trainer_utils.BestRun:
        """

        :return:
        """

        metrics = src.modelling.metrics.Metrics(id2label=self.__id2label)
        checkpoint_config = src.modelling.check.Check().__call__()
        tokenizer = src.modelling.tokenizer.Tokenizer(arguments=self.__arguments).__call__()

        # Training Arguments
        branch = 'hyperparameters'
        args = src.modelling.args.Args(
            arguments=self.__arguments, n_instances=self.__data['train'].num_rows).__call__(branch=branch)

        # Data Collator
        data_collator: transformers.DataCollatorForTokenClassification = (
            transformers.DataCollatorForTokenClassification(tokenizer=tokenizer))

        # The training object
        trainer = transformers.trainer.Trainer(
            model_init=self.__model_init, args=args, data_collator=data_collator,
            train_dataset=self.__data['train'], eval_dataset=self.__data['validation'],
            compute_metrics=metrics.exc, callbacks=[transformers.EarlyStoppingCallback(
                early_stopping_patience=self.__arguments.early_stopping_patience)])

        # Hence, hyperparameter search via ...
        best: transformers.trainer_utils.BestRun = trainer.hyperparameter_search(
            hp_space=self.__tuning.ray_hp_space, compute_objective=self.__tuning.compute_objective,
            n_trials=self.__arguments.N_TRIALS, direction='minimize', backend='ray',

            # scaling configuration
            resources_per_trial={'cpu': self.__arguments.N_CPU, 'gpu': self.__arguments.N_GPU},

            # tune configuration
            scheduler=self.__tuning.scheduler(), reuse_actors=True,

            # run configuration
            storage_path=os.path.join(self.__arguments.model_output_directory, branch, 'compute'),
            checkpoint_config=checkpoint_config,
            verbose=0, progress_reporter=self.__tuning.reporting, log_to_file=True)

        return best
