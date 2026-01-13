class ModelNotTrainedError(Exception):
    """Exception raised when attempting to use a forecasting model that has not been trained."""

    def __init__(
        self,
        method_name: str,
        extra_message: str = '',
    ):
        message = (
            f'The forecasting method "{method_name}" has not been trained yet. '
            'Please call the train() method before trying to use the trained model.'
        )
        if extra_message:
            message += f'\n{extra_message}'
        super().__init__(message)
