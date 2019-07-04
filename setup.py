# Thanks `https://github.com/pypa/sampleproject`!!

from setuptools import setup, find_packages
from os import path
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'VERSION'), 'r', encoding='utf-8') as f:
  version = f.read().strip()
with open(path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
  long_description = f.read()

setup(
  name              = 'tinynet',
  version           = version,
  description       = 'A tiny neural network library',
  long_description  = long_description,
  long_description_content_type='text/markdown',
  url               = 'https://github.com/giuse/tinynet',
  author            = 'Giuseppe Cuccu',
  author_email      = 'giuseppe.cuccu@gmail.com',
  license           = 'MIT',
  classifiers       = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
  ],
  keywords         = 'neuralnetwork machinelearning',
  packages         = find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
  python_requires  = '>=3.6, <4',
  install_requires = ['numpy'],
  project_urls={
      'Bug Reports' : 'https://github.com/giuse/tinynet/issues',
      'Source'      : 'https://github.com/giuse/tinynet/',
  },
  download_url      = f"https://github.com/giuse/tinynet/archive/{version}.tar.gz",
)
