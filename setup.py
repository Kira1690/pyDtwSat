import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pyDTWSat",
    version = "0.0.1",
    author = "Vishal D Dafada",
    author_email = "dafadavishal6@gmail.com",
    description = ("Will update description later"),
    license = "Apache 2.0",
    keywords = "Satellite image Analysis",
    url = "https://github.com/Kira1690/pyDTWSat.git",
    packages=find_packages(where='cn_tools'),
    long_description=read('README.md'),
)