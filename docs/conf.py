"""Sphinx build."""

import sys
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from os import path

# make sure one level up is on python path
sys.path.append(path.abspath(".."))

extensions = [
    "sphinx.ext.coverage",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

project = "fiblat"
version = package_version(project)
release = version

copyright = f"{datetime.now(tz=UTC).year:d} Erik Brinkman"  # noqa: A001
author = "Erik Brinkman"
