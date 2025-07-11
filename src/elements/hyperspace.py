"""Module hyperspace.py"""
import typing


class Hyperspace(typing.NamedTuple):
    """
    <b>Notes</b><br>
    ------<br>

    The hyperparameters; alongside their starting values or number spaces.<br><br>


    <b>Attributes</b><br>
    -----------<br>

    learning_rate_distribution: list[float]
        <ul><li>A list of length two, consisting of the minimum & maximum value constraints vis-à-vis
        the distribution whence learning rate values are randomly drawn from.</li></ul>
    weight_decay_distribution: list[float]
        <ul><li>A list of length two, consisting of the minimum & maximum value constraints vis-à-vis
        the distribution whence weight decay values are randomly drawn from.</li></ul>
    weight_decay_choice: list[float]
        <ul><li>A list of length one, or more, listing the weight decay values that should be considered
        during a hyperparameter search.</li></ul>
    per_device_train_batch_size: list[int]
        <ul><li>A list of length one, or more, listing the training batch size values that should be considered
        during a hyperparameter search.</li></ul><br>
    """

    learning_rate_distribution: list[float]
    weight_decay_distribution: list[float]
    weight_decay_choice: list[float]
    per_device_train_batch_size: list[int]
