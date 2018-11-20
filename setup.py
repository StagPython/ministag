from setuptools import setup

with open('README.rst') as rdm:
    README = rdm.read()

DEPENDENCIES = [
    'numpy>=1.15',
    'scipy>=1.0',
    'matplotlib>=3.0',
    'toml>=0.10',
]

setup(
    name='ministag',
    version='0.1.0',

    description='2D convection code',
    long_description=README,

    author='Stéphane Labrosse, Adrien Morison',

    python_requires='>=3.5',
    packages=['ministag'],
    entry_points={
        'console_scripts': ['stagpy = stagpy.__main__:main']
    },
    install_requires=DEPENDENCIES,
)
