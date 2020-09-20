from distutils.core import setup
import io
import os
import re
from setuptools import setup, find_packages
from pkg_resources import DistributionNotFound, get_distribution


INSTALL_REQUIRES = ["numpy>=1.11.1", "scipy", "scikit-image>=0.16.1", "imgaug>=0.4.0", "PyYAML"]

# If first not installed install second package
CHOOSE_INSTALL_REQUIRES = [("opencv-python>=4.1.1", "opencv-python-headless>=4.1.1")]


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "albumentations", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def choose_requirement(main, secondary):
    """If some version version of main requirement installed, return main,
    else return secondary.

    """
    try:
        name = re.split(r"[!<>=]", main)[0]
        get_distribution(name)
    except DistributionNotFound:
        return secondary

    return str(main)


def get_install_requirements(install_requires, choose_install_requires):
    for main, secondary in choose_install_requires:
        install_requires.append(choose_requirement(main, secondary))

    return install_requires

setup(
    name='visionlib',
    version='1.0.0',
    packages=[''],
    url='git+https://github.com/tfdeepnet/visionlib.git',
    license='MIT',
    author='Deepak',
    author_email='',
    description='pytorch cnn models for visualization'
    install_requires=get_install_requirements(INSTALL_REQUIRES, CHOOSE_INSTALL_REQUIRES),
   
)
