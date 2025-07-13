"""Module args.py"""
import os

import transformers

import src.elements.arguments as ag


class Args:
    """
    Args
    """

    def __init__(self, arguments: ag.Arguments, n_instances: int):
        """

        :param arguments:
        :param n_instances: The number of training data instances
        """

        self.__arguments = arguments
        self.__n_instances = n_instances

    def __call__(self) -> transformers.TrainingArguments:
        """

        :return:
        """

        match self.__arguments.scheduler:
            case 'PopulationBasedTraining':
                max_steps_per_epoch = self.__n_instances // (self.__arguments.TRAIN_BATCH_SIZE * self.__arguments.N_GPU)
                max_steps = int(max_steps_per_epoch * self.__arguments.EPOCHS)
            case _:
                max_steps = -1

        args = transformers.TrainingArguments(
            output_dir=self.__arguments.model_output_directory,
            report_to='tensorboard',
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=self.__arguments.LEARNING_RATE,
            weight_decay=self.__arguments.WEIGHT_DECAY,
            per_device_train_batch_size=self.__arguments.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=self.__arguments.VALID_BATCH_SIZE,
            num_train_epochs=self.__arguments.EPOCHS,
            max_steps=max_steps,
            warmup_steps=0,
            use_cpu=False,
            seed=5,
            save_total_limit=self.__arguments.save_total_limit,
            skip_memory_metrics=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            load_best_model_at_end=True,
            logging_dir=os.path.join(self.__arguments.model_output_directory, 'logs'),
            fp16=True,
            push_to_hub=False)

        return args
