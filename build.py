from distutils.command.build_ext import build_ext

from setuptools import Extension


def build(setup_kwargs):
    setup_kwargs.update(
        {
            "ext_modules": [
                Extension(
                    "zarrita.blosc",
                    ["zarrita/blosc.c"],
                    extra_compile_args=[
                        "/opt/homebrew/Cellar/c-blosc/1.21.4/lib/libblosc.a",
                        "-O3",
                    ],
                    include_dirs=["/opt/homebrew/Cellar/c-blosc/1.21.4/include"],
                )
            ],
            "cmdclass": {"build_ext": build_ext},
        }
    )
