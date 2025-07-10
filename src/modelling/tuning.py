"""Module tuning.py"""
import logging

import ray
import ray.tune
import ray.tune.schedulers as rts
import ray.tune.search.optuna as pta

import src.elements.arguments as ag
import src.elements.hyperspace as hp


class Tuning:
    """
    Class Tuning
    """

    def __init__(self, arguments: ag.Arguments, hyperspace: hp.Hyperspace):
        """

        :param arguments: A suite of values/arguments for machine learning model development
        :param hyperspace:
        """

        self.__arguments = arguments
        self.__hyperspace = hyperspace

        self.__space = {
            'learning_rate': ray.tune.uniform(lower=min(self.__hyperspace.learning_rate_distribution),
                                              upper=max(self.__hyperspace.learning_rate_distribution)),
            'weight_decay': ray.tune.uniform(lower=min(self.__hyperspace.weight_decay_distribution),
                                             upper=max(self.__hyperspace.weight_decay_distribution)),
            'per_device_train_batch_size': ray.tune.choice(self.__hyperspace.per_device_train_batch_size)}

    @staticmethod
    def compute_objective(metric):
        """

        :param metric:
        :return:
        """

        return metric['eval_loss']

    def ray_hp_space(self, trial):
        """

        :param trial: A placeholder
        :return:
        """

        logging.info(trial)

        return self.__space

    def optuna_hp_space(self, trial):

        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", min(self.__hyperspace.learning_rate_distribution),
                max(self.__hyperspace.learning_rate_distribution), log=True),
            "weight_decay": trial.suggest_float(
                "weight_decay", min(self.__hyperspace.weight_decay_distribution),
                max(self.__hyperspace.weight_decay_distribution), log=True),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", self.__hyperspace.per_device_train_batch_size),
        }

    @staticmethod
    def scheduler() ->  rts.AsyncHyperBandScheduler:
        """
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html<br>
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.AsyncHyperBandScheduler.html

        return rts.PopulationBasedTraining(
            time_attr='training_iteration',
            metric='eval_loss', mode='min',
            perturbation_interval=self.__arguments.perturbation_interval,
            hyperparam_mutations=self.__space,
            quantile_fraction=self.__arguments.quantile_fraction,
            resample_probability=self.__arguments.resample_probability)

        :return:
        """

        return rts.ASHAScheduler(
            time_attr='training_iteration', metric='eval_loss', mode='min')

    @staticmethod
    def algorithm() -> pta.OptunaSearch:
        """
        ray.tune.search.optuna.OptunaSearch(spae=, metric='eval_loss', mode='min')

        :return:
        """

        return pta.OptunaSearch(metric='eval_loss', mode='min')

    @staticmethod
    def reporting():
        """

        Notes<br>
        ------<br>

        <a href="https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CLIReporter.html">ray.tune.CLIReporter</a><br><br>

        This prints to the console if the verbose setting of
        <a href="https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/
        trainer#transformers.Trainer.hyperparameter_search"> hyperparameter_search()</a> is > 0. The
        hyperparameter_search() function accepts the
        <a href="https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html">RunConfig</a> parameters.

        :return:
        """

        return ray.tune.CLIReporter(
            parameter_columns=['learning_rate', 'weight_decay', 'per_device_training_batch_size'],
            metric_columns=['eval_loss', 'precision', 'recall', 'f1'])
