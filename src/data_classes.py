import gzip
import sys
from typing import Self

import brian2.only as b2
import dill


class BaseDataClass:
    def save(self, filename):
        assert filename.endswith(".pkl.gz")
        with gzip.open(filename, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename) -> Self:
        assert filename.endswith(".pkl.gz")
        if b2.prefs.codegen.runtime.cython.cache_dir not in sys.path:
            sys.path.append(b2.prefs.codegen.runtime.cython.cache_dir)
        with gzip.open(filename, "rb") as f:
            return dill.load(f)
