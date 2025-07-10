import ray.train.torch
import ray.train
import ray.tune

import src.modelling.check
import src.modelling.intelligence
import src.modelling.tuning
import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.s3_parameters as s3p


class Steps:

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
        {'cpu': self.__arguments.N_CPU, 'gpu': self.__arguments.N_GPU}
        :return:
        """

        intelligence = src.modelling.intelligence.Intelligence(
            s3_parameters=self.__s3_parameters, arguments=self.__arguments, hyperspace=self.__hyperspace)

        checkpoint_config = src.modelling.check.Check().__call__()

        tuning = src.modelling.tuning.Tuning(arguments=self.__arguments, hyperspace=self.__hyperspace)

        trainer: ray.train.torch.TorchTrainer = ray.train.torch.TorchTrainer(
            intelligence.train_func,
            scaling_config=ray.train.ScalingConfig(
                resources_per_worker={'CPU': self.__arguments.N_CPU, 'GPU': self.__arguments.N_GPU},
                use_gpu=True, num_workers=self.__arguments.N_GPU),
            run_config=ray.train.RunConfig(checkpoint_config=checkpoint_config)
        )

        ray.tune.Tuner(
            trainer,
            param_space={"train_loop_config": tuning.space}
        )






