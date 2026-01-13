from lh_v2.util import BaseParamsModel


class OutputParams(BaseParamsModel):
    """
    Configuration parameters for output settings.

    Attributes
    ----------
    b_use_fit_forecasts : bool
        Indicates whether to use fit forecasts. Default is True.
    account_file_format : str
        The file format for account data. Default is "parquet".
    csv_delimiter : str
        The delimiter to use for CSV files. Default is ",".
    b_split_drivers : bool
        Indicates whether to split drivers. Default is False.
    """

    b_finish_driver_analysis_file: bool = True
    b_use_fit_forecasts: bool = True
    account_file_format: str = 'parquet'
    csv_delimiter: str = ','
    b_split_drivers: bool = False
    b_test_outputs: bool = False
