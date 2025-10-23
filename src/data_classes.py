from typing import Self

from src.helpers import load_pkl_gz, save_pkl_gz


class BaseDataClass:
    def save(self, filename):
        save_pkl_gz(self, filename)

    @classmethod
    def load(cls, filename) -> Self:
        return load_pkl_gz(filename)
