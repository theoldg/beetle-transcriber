from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict
import yaml


class Config(BaseModel):
    model_config = ConfigDict(strict=True)

    @classmethod
    def load_from_yaml(cls, path: Path | str) -> Self:
        path = Path(path)
        content = path.read_text()
        parsed_dict = yaml.safe_load(content)
        return cls(**parsed_dict)

