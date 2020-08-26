"""
Setup script for CoalPick
"""
import glob
from os.path import join, exists, isdir
from pathlib import Path

from setuptools import setup

# get path references
here = Path(__file__).absolute().parent
readme_path = here / "README.md"
# get requirement paths
package_req_path = here / "requirements.txt"
version_file = here / "CoalPick" / "version.py"


def find_packages(base_dir="."):
    """ setuptools.find_packages wasn't working so I rolled this """
    out = []
    for fi in glob.iglob(join(base_dir, "**", "*"), recursive=True):
        if isdir(fi) and exists(join(fi, "__init__.py")):
            out.append(fi)
    out.append(base_dir)
    return out


def get_version(version_path):
    """Extract the version from the version file."""
    __version__ = None
    with Path(version_path).open() as fi:
        for line in fi.readline():
            if not line.startswith("__version__"):  # Only find version token
                continue
        content = fi.read().split("=")[-1].strip()
        __version__ = content.replace('"', "").replace("'", "")
    assert __version__ is not None, "No version found!"
    return __version__


def read_requirements(path, skip_if_missing=False):
    """ Read a requirements.txt file, return a list. """
    path = Path(path)
    if not path.exists() and skip_if_missing:
        return []
    with Path(path).open("r") as fi:
        lines = fi.readlines()
    # remove any line comments
    return [x for x in lines if not x.startswith("#")]


def load_file(path):
    """ Load a file into memory. """
    with Path(path).open() as w:
        contents = w.read()
    return contents


# --- get sub-packages


requires = read_requirements(package_req_path)

setup(
    name="CoalPick",
    version=get_version(version_file),
    description="Example code for picking phases with CNN (Johnson et al., 2020)",
    long_description=load_file(readme_path),
    author="Sean Johnson ",
    author_email="sjohnson10@zagmail.gonzaga.edu",
    url="https://github.com/sjohnson5/SAMples",
    packages=find_packages("CoalPick"),
    package_dir={"CoalPick": "CoalPick"},
    license="GNU Lesser General Public License v3.0 or later (LGPLv3.0+)",
    zip_safe=False,
    keywords="CNN",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    test_suite="tests",
    install_requires=read_requirements(package_req_path),
    python_requires=">=3",
)
