from distutils.core import setup
setup(
  name = 'tinynet',
  packages = ['tinynet'],
  version = '0.0.0',
  license='MIT',
  description = 'A tiny neural network library',
  author = 'Giuseppe Cuccu',
  author_email = 'giuseppe.cuccu@gmail.com',
  url = 'https://github.com/giuse/tinynet',
  download_url = 'https://github.com/giuse/tinynet/archive/v_01.tar.gz',
  keywords = ['neural network'],
  install_requires=[
          'numpy'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
