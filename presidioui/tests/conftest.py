from functools import wraps
import os
import tempfile
from pathlib import Path
import shutil


import pytest


def _path_to_tests():
    return Path(__file__).resolve().parent.parent / "tests"


def fixture_tmp_dir(source, **kwargs):
    """
    A decorator to create fixtures that copy files into a temporary folder
    """

    def decorator(function):
        @wraps(function)
        def wrapper():
            old = os.getcwd()
            tmp_dir = tempfile.mkdtemp()
            tmp = Path(tmp_dir, "content")
            # we have to add extra folder content/, otherwise copytree
            # complains
            shutil.copytree(str(source), str(tmp))
            os.chdir(str(tmp))
            yield tmp

            os.chdir(old)
            shutil.rmtree(tmp_dir)

        return pytest.fixture(wrapper, **kwargs)

    return decorator


@fixture_tmp_dir(_path_to_tests() / "assets")
def tmp_assets():
    pass


@pytest.fixture
def tmp_empty(tmp_path):
    """
    Create temporary path using pytest native fixture,
    them move it, yield, and restore the original path
    """
    old = os.getcwd()
    os.chdir(str(tmp_path))
    yield str(Path(tmp_path).resolve())
    os.chdir(old)
