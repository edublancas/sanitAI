from pathlib import Path
import sys
from copy import copy
from inspect import getmembers
import importlib
import os
from contextlib import contextmanager


@contextmanager
def add_to_sys_path(path):
    """Add the given path to sys.path for the duration of the context"""
    path = os.path.abspath(path)
    sys.path.insert(0, path)

    try:
        yield
    finally:
        sys.path.remove(path)


class BaseSchema:
    """A base object to define a schema for settings validation"""

    @staticmethod
    def _get_public_attributes(obj):
        return {k.upper(): v for k, v in obj.__dict__.items() if not k.startswith("_")}

    @classmethod
    def _validate(cls, settings):
        validators = cls._get_public_attributes(cls)
        validators_docs = {k: v.__doc__ for k, v in validators.items()}

        missing = sorted(set(validators) - set(settings))

        if missing:
            missing_with_docs = {
                k: v for k, v in validators_docs.items() if k in missing
            }

            formatted_error = "\n".join(
                [f"{k}: {v}" for k, v in missing_with_docs.items()]
            )

            raise RuntimeError(
                f"Error validating settings, missing:\n\n{formatted_error}"
            )

        unexpected = sorted(set(settings) - set(validators))

        if unexpected:
            formatted_error = "\n".join([k for k in unexpected])

            raise RuntimeError(
                f"Error validating settings, unexpected:\n\n{formatted_error}"
            )

        matched = set(validators) & set(settings)

        for match in matched:
            try:
                validators[match](settings[match])
            except Exception as e:
                raise RuntimeError(
                    f"Error validating settings, the validator for {match} "
                    f"failed: {str(e)}"
                ) from e


class Schema(BaseSchema):
    """The schema that validates the settings

    Notes
    -----
    To register a new setting, add a new method to this class. The method name
    must match the setting name in uppercase. The method docstring will be
    used to display the error message if the setting is missing or invalid.
    The body of the function can raise exceptions to validate the setting.
    """

    @staticmethod
    def path_to_here(value):
        """Path to the directory where the settings.py file is located"""
        pass

    @staticmethod
    def db_uri(value):
        """URI to the database"""
        pass

    @staticmethod
    def model_reasoning(value):
        """The model to use for reasoning"""
        pass

    @staticmethod
    def model_default(value):
        """The model to use for default"""
        pass

    @staticmethod
    def secret_key(value):
        """The secret key for the flask application"""
        pass


class BaseSettings:
    """A base object to load settings from a settings.py file"""

    SCHEMA = None

    def __init__(self) -> None:
        self._path_to_settings, _ = find_file_recursively("settings.py")

        if not self._path_to_settings:
            raise RuntimeError("No settings.py file found")

        self._load()

    def _load(self):
        with add_to_sys_path(self._path_to_settings.parent):
            module = importlib.import_module("settings")

        del sys.modules["settings"]

        self._settings = {k: v for k, v in getmembers(module) if k.upper() == k}
        self.SCHEMA._validate(self._settings)

        for k, v in self._settings.items():
            setattr(self, k, v)

    def to_dict(self):
        return copy(self._settings)

    def to_environ(self):
        """Set all settings as environment variables"""
        for k, v in self.to_dict().items():
            os.environ[k] = str(v)


class Settings(BaseSettings):
    """
    The settings object used to load settings from settings.py. It validates
    with the Schema class
    """

    SCHEMA = Schema


def find_file_recursively(name, max_levels_up=10, starting_dir=None):
    """
    Find a file by looking into the current folder and parent folders,
    returns None if no file was found otherwise pathlib.Path to the file

    Parameters
    ----------
    name : str
        Filename

    Returns
    -------
    path : str
        Absolute path to the file
    levels : int
        How many levels up the file is located
    """
    current_dir = starting_dir or os.getcwd()
    current_dir = Path(current_dir).resolve()
    path_to_file = None
    levels = None

    for levels in range(max_levels_up):
        current_path = Path(current_dir, name)

        if current_path.exists():
            path_to_file = current_path.resolve()
            break

        current_dir = current_dir.parent

    return path_to_file, levels
