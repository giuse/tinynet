from distutils.core import setup
from subprocess import run, PIPE

version = run(['git', 'describe'], stdout=PIPE).stdout.decode('utf8').strip()

setup(
  name         = 'tinynet',
  packages     = ['tinynet'],
  version      = version,
  license      = 'MIT',
  description  = 'A tiny neural network library',
  author       = 'Giuseppe Cuccu',
  author_email = 'giuseppe.cuccu@gmail.com',
  url          = 'https://github.com/giuse/tinynet',
  download_url = f"https://github.com/giuse/tinynet/archive/{version}.tar.gz",
  keywords     = ['neural network'],
  install_requires = [
          'numpy'
      ],
  classifiers  = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
