"""Module mappings.py"""
import transformers

import datasets
import numpy as np
import torch


class Mappings:
    """
    Computes tokens according to the expectations of the T5 architecture
    """

    def __init__(self, tokenizer: transformers.models.t5.tokenization_t5_fast.T5TokenizerFast, _id2label) -> None:
        """

        :param tokenizer:
        :param _id2label:
        """

        self.__id2label = _id2label
        self.__tokenizer = tokenizer

    def injective(self, _targets: list) -> list:
        """
        If the text_target of a tokenizer expects text that it can tokenize, i.e., not codes, then
        let the raw labels be the text_target

        :param _targets: The targets of the split strings/words of a distinct sentence
        :return:
        """

        return list(map(lambda y: self.__id2label[y], _targets))

    @staticmethod
    def __surjective(_tags: list, _packets: list) -> list:
        """

        :param _tags: Used to get the tag associated with a token
        :param _packets: A padded list of a sentence's strings/words position identifiers per token
        :return:
        """

        _surjective = list(map(lambda x: int(_tags[x]) if x is not None else int(-100), _packets))

        return _surjective

    def bijective(self, tags: list[list[int]], packets: transformers.tokenization_utils_base.BatchEncoding) -> torch.Tensor:
        """

        :param tags: The corresponding tag codes of packets.
        :param packets: Each batch index denotes the location of a list of strings/words that
                        make-up a sentence; a location amongst a list of lists.
        :return:
        """

        indices = np.arange(len(tags))

        vectors = list(
            map(lambda i: self.__surjective(tags[i], packets.word_ids(batch_index=i)),
                indices)
        )

        return torch.tensor(vectors, dtype=torch.int64)

    def exc(self, feeds: datasets.arrow_dataset.Dataset) -> transformers.tokenization_utils_base.BatchEncoding:
        """

        :param feeds: Either a training, validation, or testing Dataset.
        """

        # A list of lists; each distinct list consists of the split strings/words of a sentence.
        samples = feeds['tokens']

        # A list of lists
        targets = list(map(self.injective, feeds['fine_ner_tags']))

        # Tokenization
        packets = self.__tokenizer(
            samples, text_target=targets, is_split_into_words=True, padding='max_length', truncation=True,
            return_tensors='pt', return_token_type_ids=True)

        # The original labels/targets
        vectors = self.bijective(tags=feeds['fine_ner_tags'], packets=packets)

        packets['initial'] = packets['labels']
        packets['labels'] = vectors

        return packets
