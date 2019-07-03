from distutils.core import setup
from subprocess import run, PIPE

version = run(['git', 'describe'], stdout=PIPE).stdout.decode('utf8').strip()
with open('README.md', encoding='utf-8') as f: long_description = f.read()

setup(
  name              = 'tinynet',
  packages          = ['tinynet'],
  version           = version,
  license           = 'MIT',
  description       = 'A tiny neural network library',
  long_description  = long_description,
  author            = 'Giuseppe Cuccu',
  author_email      = 'giuseppe.cuccu@gmail.com',
  url               = 'https://github.com/giuse/tinynet',
  download_url      = f"https://github.com/giuse/tinynet/archive/{version}.tar.gz",
  keywords          = 'neuralnetwork machinelearning',
  install_requires  = ['numpy'],
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
)
