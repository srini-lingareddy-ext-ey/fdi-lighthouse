from typing import Literal

from pydantic import Field

from lh_v2.util import BaseParamsModel


class OutputParams(BaseParamsModel):
    b_save_info: bool = False
    unique_prefix: str = Field(default_factory=lambda: '')
    table_file_format: Literal['csv', 'parquet'] = Field(default_factory=lambda: 'csv')
    other_file_format: Literal['json'] = Field(default_factory=lambda: 'json')
    save_dir_name: str = Field(default_factory=lambda: '')

    def get_save_dir_name(self) -> str:
        assert self.save_dir_name, (
            'save_dir_name not properly set after parsing config.'
        )
        return self.save_dir_name
