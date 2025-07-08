import os
import transformers

import src.elements.arguments as ag

class Args:

    def __init__(self, arguments: ag.Arguments):

        self.__arguments = arguments

    def __call__(self) -> transformers.TrainingArguments:

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
            max_steps=-1,
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

