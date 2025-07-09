"""Module ctrc.py"""
import ray.train
import transformers
import transformers.trainer_callback as ttc


# noinspection SpellCheckingInspection
class CTRC(ttc.TrainerCallback):
    """
    CustomTrainReportCallback.  If<br>

    "... Ray Train’s default RayTrainReportCallback is [insufficient] ... implement a callback ...".  For example,
    this "... implementation ... collects [the] latest metrics and reports on checkpoint save."
    """

    def __init__(self):
        """
        Constructor
        """

        super().__init__()
        self.metrics = {}

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """
        Log is called on evaluation step and logging step.

        :param args:
        :param state:
        :param control:
        :param model:
        :param logs:
        :param kwargs:
        :return:
        """

        self.metrics.update(logs)

    def on_save(self, args, state, control, **kwargs):
        """
        Event called after a checkpoint save.

        :param args:
        :param state:
        :param control:
        :param kwargs:
        :return:
        """

        checkpoint = None
        if ray.train.get_context().get_world_rank() == 0:

            # Build a Ray Train Checkpoint from the latest checkpoint
            path = transformers.trainer.get_last_checkpoint(args.output_dir)
            checkpoint = ray.train.Checkpoint.from_directory(path=path)

        # Report to Ray Train with up-to-date metrics
        ray.train.report(metrics=self.metrics, checkpoint=checkpoint)

        # Metrics buffer flushing
        self.metrics = {}
