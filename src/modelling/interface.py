"""Module interface.py"""
import ast
import logging
import os

import boto3
import transformers

import src.elements.arguments as ag
import src.elements.hyperspace as hp
import src.elements.master as mr
import src.modelling.architecture
import src.modelling.convergence
import src.modelling.tokenization
import src.valuate.interface


# noinspection DuplicatedCode
class Interface:
    """
    Layer
    """

    def __init__(self, connector: boto3.session.Session, arguments: ag.Arguments, hyperspace: hp.Hyperspace):
        """

        :param connector: A boto3 session instance, it retrieves the developer's <default> Amazon
                          Web Services (AWS) profile details, which allows for programmatic interaction with AWS.
        :param arguments:
        :param hyperspace:
        """

        self.__connector = connector
        self.__arguments = arguments
        self.__hyperspace = hyperspace

    def exc(self, master: mr.Master):
        """

        :param master:
        :return:
        """

        # Tokenization
        master = src.modelling.tokenization.Tokenization(arguments=self.__arguments).exc(master=master)

        # Best: Hyperparameters
        best = src.modelling.architecture.Architecture(
            arguments=self.__arguments, hyperspace=self.__hyperspace, master=master).train_func(branch='hyperparameters')
        logging.info(best)
        logging.info(best.run_summary)
        logging.info(best.hyperparameters)

        # Hence, update the modelling variables
        self.__arguments = self.__arguments._replace(
            LEARNING_RATE=best.hyperparameters.get('learning_rate'),
            WEIGHT_DECAY=best.hyperparameters.get('weight_decay'),
            TRAIN_BATCH_SIZE=best.hyperparameters.get('per_device_train_batch_size'))

        # Additionally, prepare the artefacts storage area for the best model, vis-à-vis best hyperparameters
        # set, and save a checkpoint at the optimal training point only by setting save_total_limit = 1.
        self.__arguments = self.__arguments._replace(
            EPOCHS=2*self.__arguments.EPOCHS, save_total_limit=1)

        # Optimal Model
        branch = 'optimal'
        model: transformers.Trainer = src.modelling.convergence.Convergence(
            arguments=self.__arguments, master=master).__call__(branch=ast.literal_eval(branch))

        model.save_model(output_dir=os.path.join(self.__arguments.model_output_directory, branch, 'model'))

        interface = src.valuate.interface.Interface(
            connector=self.__connector, model=model, id2label=master.id2label, arguments=self.__arguments)
        interface.exc(blob=master.data['validation'], branch=branch, stage='validation')
        interface.exc(blob=master.data['test'], branch=branch, stage='test')
