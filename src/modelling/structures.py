"""Module structures.py"""
import logging
import os

import ray
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
class Structures:
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

        # For the tags, and the datasets.DatasetDict
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

    def train_func(self, config: dict) -> transformers.trainer.Trainer:
        """

        :return:
        """

        metrics = src.modelling.metrics.Metrics(id2label=self.__id2label)

        # Data
        data = self.__bytes.data()
        train_dataset = ray.data.from_huggingface(data['train'])
        eval_dataset = ray.data.from_huggingface(data['validation'])

        # Training Arguments
        t_batch = config.get('per_device_train_batch_size', self.__arguments.TRAIN_BATCH_SIZE)
        max_steps_per_epoch = self.__arguments.N_INSTANCES // (t_batch * self.__arguments.N_GPU)
        max_steps = int(max_steps_per_epoch * self.__arguments.EPOCHS)

        args = transformers.TrainingArguments(
            output_dir=self.__arguments.model_output_directory, report_to='tensorboard',
            eval_strategy='epoch', save_strategy='epoch',
            learning_rate=config.get('learning_rate', self.__arguments.LEARNING_RATE),
            weight_decay=config.get('weight_decay', self.__arguments.WEIGHT_DECAY),
            per_device_train_batch_size=t_batch,
            per_device_eval_batch_size=self.__arguments.VALID_BATCH_SIZE,
            num_train_epochs=self.__arguments.EPOCHS, max_steps=max_steps,
            warmup_steps=0, use_cpu=False, seed=5,
            save_total_limit=self.__arguments.save_total_limit, skip_memory_metrics=True,
            metric_for_best_model='eval_loss', greater_is_better=False, load_best_model_at_end=True,
            logging_dir=os.path.join(self.__arguments.model_output_directory, 'logs'), fp16=True, push_to_hub=False)

        # The training object
        trainer = transformers.trainer.Trainer(
            model_init=self.__model_init, args=args,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            compute_metrics=metrics.exc, callbacks=[transformers.EarlyStoppingCallback(
                early_stopping_patience=self.__arguments.early_stopping_patience)])

        # https://docs.ray.io/en/latest/train/getting-started-transformers.html#report-checkpoints-and-metrics
        # trainer.add_callback(rtht.RayTrainReportCallback())
        # trainer = rtht.prepare_trainer(trainer=trainer)

        return trainer.train()
