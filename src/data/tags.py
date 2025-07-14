import typing
import datasets

class Tags:
    """
    Tags
    """

    def __init__(self):
        pass

    @staticmethod
    def exc(feed: datasets.arrow_dataset.Dataset) -> typing.Tuple[dict, dict]:
        """

        :param feed: One of the datasets.Dataset of the raw data's datasets.DatasetDict
        :return:
        """

        values: datasets.Sequence = feed.features['fine_ner_tags']
        id2label = dict(enumerate(values.feature.names))
        label2id = {value: key for key, value in id2label.items()}

        return id2label, label2id
