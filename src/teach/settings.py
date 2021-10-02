# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from functools import lru_cache
from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    AI2THOR_BASE_DIR: str = Path.home().joinpath(".ai2thor").as_posix()
    AI2THOR_USE_LOCAL_EXE: bool = False
    GUNICORN_LOGGING: bool = False

    class Config:
        env_prefix = "TEACH_"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
