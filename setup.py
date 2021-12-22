from setuptools import setup

with open('README.rst') as rdm:
    README = rdm.read()

DEPENDENCIES = [
    'h5py>=3.0',
    'numpy>=1.19',
    'scipy>=1.5',
    'matplotlib>=3.3',
    'toml>=0.10.2',
]

setup(
    name='ministag',
    version='0.1.0',

    description='2D convection code',
    long_description=README,

    author='Stéphane Labrosse, Adrien Morison',

    python_requires='>=3.7',
    packages=['ministag'],
    entry_points={
        'console_scripts': ['ministag = ministag.__main__:main']
    },
    install_requires=DEPENDENCIES,
)
