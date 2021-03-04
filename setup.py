from setuptools import find_packages, setup

with open("README.md", "r") as f:
  long_description = f.read()

setup(
    name='fatima',
    version='0.0.1',
    description="Fatima Fellowship Coding challenges",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Amr Kayid",
    url="https://github.com/AmrMKayid/fatima",
    packages=find_packages(),
    install_requires=[
        "loguru",
        "torch",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
