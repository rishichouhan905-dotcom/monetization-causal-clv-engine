import os
import gc
import pytest

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


@pytest.fixture(autouse=True)
def run_gc():
    yield
    gc.collect()
