#!/usr/bin/env python

from setuptools import setup
import subprocess
from pathlib import Path
import os


def write_version_file(version):
    """ Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file (relative to the src package directory)
    """
    version_file = Path("src") / ".version"

    try:
        git_log = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%h %ai"]
        ).decode("utf-8")
        git_diff = (
            subprocess.check_output(["git", "diff", "."])
            + subprocess.check_output(["git", "diff", "--cached", "."])
        ).decode("utf-8")
    except subprocess.CalledProcessError as exc:  # git calls failed
        # we already have a version file, let's use it
        if version_file.is_file():
            return version_file.name
        # otherwise error out
        exc.args = (
            "unable to obtain git version information, and {} doesn't "
            "exist, cannot continue ({})".format(version_file, str(exc)),
        )
        raise
    else:
        git_version = "{}: ({}) {}".format(
            version, "UNCLEAN" if git_diff else "CLEAN", git_log.rstrip()
        )
        print("parsed git version info as: {!r}".format(git_version))

    with open(version_file, "w") as f:
        print(git_version, file=f)
        print("created {}".format(version_file))

    return version_file.name


def get_long_description():
    """ Finds the README and reads in the description """
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.md")) as f:
        long_description = f.read()
    return long_description


# get version info from __init__.py
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()
    return filecontents


VERSION = '0.2.0'
version_file = write_version_file(VERSION)
long_description = get_long_description()

setup(name='kookaburra',
      description='Shapelet fitting for pulsars',
      long_description=long_description,
      url='https://github.com/GregoryAshton/kookaburra',
      author='Greg Ashton',
      author_email='gregory.ashton@ligo.org',
      license="MIT",
      version=VERSION,
      packages=['kookaburra'],
      package_dir={'kookaburra': 'src'},
      package_data={'kookaburra': [version_file]},
      python_requires='>3.6',
      install_requires=['bilby'],
      entry_points={'console_scripts':
                    ['kb_single_pulse=kookaburra.single_pulse:main',
                     'kb_create_database=kookaburra.database:main'
                     ]
                    },
      classifiers=[
          "Programming Language :: Python :: 3.7",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent"])
