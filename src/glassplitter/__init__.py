from importlib import metadata

from glassplitter.tokenizer import Tokenizer  # noqa: F401

__version__: str = str(metadata.version("glassplitter"))
