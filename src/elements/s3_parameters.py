"""
This is data type S3Parameters
"""
import typing


class S3Parameters(typing.NamedTuple):
    """
    The data type class ⇾ S3Parameters

    Attributes
    ----------
    region_name : str
      The Amazon Web Services region code.

    location_constraint : str
      The region code of the region that the data is limited to.

    internal : str
      An Amazon S3 (Simple Storage Service) bucket.

    path_internal_data : str
      The data bucket path.

    path_internal_artefacts : str
      The bucket path of the model development artefacts.

    external : str
      An Amazon S3 (Simple Storage Service) bucket.

    configurations : str
      An Amazon S3 (Simple Storage Service) bucket.
    """

    region_name: str
    location_constraint: str
    internal: str
    path_internal_data: str
    path_internal_artefacts: str
    external: str
    configurations : str
