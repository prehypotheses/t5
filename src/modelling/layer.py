import logging

import src.data.interface
import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.s3_parameters as s3p
import src.modelling.args
import src.modelling.check
import src.modelling.metrics
import src.modelling.structures
import src.modelling.tokenizer
import src.modelling.tuning


# noinspection DuplicatedCode
class Layer:
    """
    Layer
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

    def exc(self):
        """
        
        :return:
        """

        checkpoint_config = src.modelling.check.Check().__call__()

        # The tuning objects for model training/development
        tuning = src.modelling.tuning.Tuning(arguments=self.__arguments, hyperspace=self.__hyperspace)

        trainer = src.modelling.structures.Structures(
            s3_parameters=self.__s3_parameters, arguments=self.__arguments, hyperspace=self.__hyperspace).train_func(
            config=tuning.space)

        # Hence, hyperparameter search via ...
        # Re-design: https://github.com/huggingface/transformers/blob/main/docs/source/en/hpo_train.md
        best = trainer.hyperparameter_search(
            hp_space=tuning.ray_hp_space, compute_objective=tuning.compute_objective,
            n_trials=self.__arguments.N_TRIALS, direction=['minimize', 'minimize', 'maximize'], backend='ray',
            resources_per_trial={'cpu': self.__arguments.N_CPU, 'gpu': self.__arguments.N_GPU},
            storage_path=self.__arguments.storage_path,
            scheduler=tuning.scheduler(), reuse_actors=True,
            checkpoint_config=checkpoint_config,
            verbose=0, progress_reporter=tuning.reporting, log_to_file=True)

        return best
