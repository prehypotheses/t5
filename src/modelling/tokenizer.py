"""Module tokenizer"""
import transformers

import src.elements.arguments as ag


class Tokenizer:
    """
    Class Tokenizer: T5
        https://research.google/blog/exploring-transfer-learning-with-t5-the-text-to-text-transfer-transformer/
        https://arxiv.org/abs/1910.10683
    """

    def __init__(self, arguments: ag.Arguments):
        """

        :param arguments: A suite of values/arguments for machine learning model development.<br>
        """

        self.__arguments = arguments

    def __call__(self) -> transformers.tokenization_utils_base.PreTrainedTokenizerBase:
        """
        https://huggingface.co/docs/transformers/v4.53.1/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained

        :return:
        """

        # Tokenizer
        return transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.__arguments.pretrained_model_name,
            clean_up_tokenization_spaces=True)
