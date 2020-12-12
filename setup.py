from setuptools import setup

requirements = [
    "fsspec",
    "numpy",
    "numcodecs",
]

setup(
    name="zarrita",
    version="0.1.0",
    url="https://github.com/alimanfoo/zarrita",
    author="Alistair Miles",
    author_email="alimanfoo@googlemail.com",
    description=(
        "A minimal, exploratory implementation "
        "of the Zarr version 3.0 core protocol"
    ),
    py_modules=["zarrita"],
    install_requires=requirements,
)
