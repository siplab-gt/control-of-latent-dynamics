from typing import Self
import sys

import brian2.only as b2
import dill


class BaseDataClass:
    def save(self, filename):
        with open(filename, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename) -> Self:
        if b2.prefs.codegen.runtime.cython.cache_dir not in sys.path:
            sys.path.append(b2.prefs.codegen.runtime.cython.cache_dir)
        with open(filename, "rb") as f:
            return dill.load(f)
