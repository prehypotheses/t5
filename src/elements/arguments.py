"""Module arguments.py"""
import typing


class Arguments(typing.NamedTuple):
    """

    Attributes<br>
    ----------<br>

    MAX_LENGTH: <b>int</b>  The maximum number of tokens<br>
    TRAIN_BATCH_SIZE : <b>int</b> The batch size for the training stage<br>
    VALID_BATCH_SIZE : <b>int</b> The batch size for the validation stage<br>
    TEST_BATCH_SIZE : <b>int</b> The batch size for the testing stage<br>
    EPOCHS : <b>int</b> The number of epochs<br>
    LEARNING_RATE : <b>float</b> The learning rate<br>
    WEIGHT_DECAY : <b>float</b>    Weight decay<br>
    MAX_GRADIENT_NORM : <b>int</b> The maximum gradient norm<br>
    N_TRAIN : <b>int</b>   The number of training instances<br>
    N_VALID : <b>int</b>   The number of validation instances<br>
    N_TEST : <b>int</b>    The number of testing instances<br>
    N_CPU: <b>int</b>  An initial number of central processing units for computation<br>
    N_GPU: <b>int</b>  The number of graphics processing units<br>
    N_TRIALS: <b>int</b>   Hyperparameters search trials<br>
    N_INSTANCES: <b>int</b> The total number of data instances to use for training, validating, and testing.<br>
    save_total_limit: <b>int</b>
        <a href="https://huggingface.co/docs/setfit/reference/trainer#setfit.TrainingArguments.save_total_limit"
        target="_blank">
        The maximum # of checkpoints that will be retained.</a><br>
    early_stopping_patience: <b>int</b>
        <a
        href="https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/callback#transformers.EarlyStoppingCallback">
        For EarlyStoppingCallback</a>: <i>in re</i> transformers.Trainer.<br>
    perturbation_interval: <b>float</b>
        <a href="https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html" target="_blank">
        Population Based Training</a><br>
    quantile_fraction: <b>float</b>
        <a href="https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html" target="_blank">
        Population Based Training</a><br>
    resample_probability: <b>float</b>
        <a href="https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html" target="_blank">
        Population Based Training</a><br>
    task : <b>str</b> The type of task the model is being trained for<br>
    pretrained_model_name : <b>str</b> The name of the pre-trained model that will be fine-tuned<br>
    architecture : <b>str</b> A name that identifies the underlying pre-trained model.<br>
    model_output_directory: <b>str</b>
    """

    MAX_LENGTH: int
    TRAIN_BATCH_SIZE: int
    VALID_BATCH_SIZE: int
    TEST_BATCH_SIZE: int
    EPOCHS: int
    LEARNING_RATE: float
    WEIGHT_DECAY: float
    MAX_GRADIENT_NORM: int
    N_TRAIN: int
    N_VALID: int
    N_TEST: int
    N_CPU: int
    N_GPU: int
    N_TRIALS: int
    N_INSTANCES: int
    save_total_limit: int
    early_stopping_patience: int
    perturbation_interval: int
    quantile_fraction: float
    resample_probability: float
    task: str
    pretrained_model_name: str
    architecture: str
    model_output_directory: str
