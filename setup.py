import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stonybook",
    version="1.0.0",
    author="Data Science Lab",
    author_email="cpethe@cs.stonybrook.edu",
    description="Stony Book Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sbu-dsl/stonybook",
    packages=setuptools.find_packages(),
    package_data={'stonybook': ['stonybook/schemas/*']},
    include_package_data=True,
    python_requires='>=3.7',
)
