"""Module steps.py"""
import ray.train
import ray.train.torch
import ray.tune

import src.data.interface
import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.s3_parameters as s3p
import src.modelling.check
import src.modelling.intelligence
import src.modelling.tuning


class Steps:
    """

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

        # For the tags, and the datasets.DatasetDict
        self.__bytes = src.data.interface.Interface(s3_parameters=s3_parameters)


    def exc(self):
        """

        :return:
        """

        # Data
        data = self.__bytes.data()
        train = ray.data.from_huggingface(data['train'])
        validation = ray.data.from_huggingface(data['validation'])

        # Update self.__arguments
        self.__arguments = self.__arguments._replace(N_INSTANCES=data['train'].num_rows)

        # transformers.trainer.Trainer
        id2label, label2id = self.__bytes.tags()
        intelligence = src.modelling.intelligence.Intelligence(
            s3_parameters=self.__s3_parameters, arguments=self.__arguments,
            hyperspace=self.__hyperspace, id2label=id2label, label2id=label2id)

        # ray.train.torch.TorchTrainer
        checkpoint_config = src.modelling.check.Check().__call__()
        trainer: ray.train.torch.TorchTrainer = ray.train.torch.TorchTrainer(
            intelligence.train_func,
            scaling_config=ray.train.ScalingConfig(
                resources_per_worker={'CPU': self.__arguments.N_CPU, 'GPU': self.__arguments.N_GPU},
                use_gpu=True, num_workers=self.__arguments.N_GPU),
            run_config=ray.train.RunConfig(checkpoint_config=checkpoint_config),
            datasets={'train': train, 'eval': validation}
        )

        # ray.tune.Tuner
        tuning = src.modelling.tuning.Tuning(arguments=self.__arguments, hyperspace=self.__hyperspace)
        tuner = ray.tune.Tuner(
            trainer,
            param_space={"train_loop_config": tuning.space},
            tune_config=ray.tune.TuneConfig(
                metric='eval_loss', mode='min', scheduler=tuning.scheduler())
        )

        grid = tuner.fit()

        return grid
