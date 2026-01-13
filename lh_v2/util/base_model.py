import pydantic as pyd


class BaseModel(pyd.BaseModel):
    model_config = pyd.ConfigDict(
        use_attribute_docstrings=True,
    )


class BaseParamsModel(pyd.BaseModel):
    model_config = pyd.ConfigDict(
        use_attribute_docstrings=True,
    )
