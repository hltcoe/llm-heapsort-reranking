import os

import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


# from https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), "rt") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="llm-heapsort-reranking",
    version=get_version("llm_heapsort_reranking/__init__.py"),
    author="HLTCOE",
    author_email="TODO",
    description="TODO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hltcoe/llm-heapsort-reranking",
    packages=setuptools.find_packages(),
    install_requires=["TODO"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    # TODO
    #    entry_points={
    #        "console_scripts": [
    #            "bsparse=bsparse.cli:__main__",
    #        ]
    #    },
)
