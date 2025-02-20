from setuptools import setup

# read the contents of your README file
from pathlib import Path


with open(Path(__file__).parent.joinpath("OpenDHW", "__init__.py"), "r") as file:
    for line in file.readlines():
        if line.startswith("__version__"):
            VERSION = line.replace("__version__", "").split("=")[1].strip().replace("'", "").replace('"', '')

setup(
    name="OpenDHW",
    version=VERSION,
    description="Tool for generating domestic hot water profiles",
    url="https://github.com/RWTH-EBC/OpenDHW",
    author="RWTH Aachen University, E.ON Energy Research Center, "
    "Institute of Energy Efficient Buildings and Indoor Climate",
    author_email="ebc-tools@eonerc.rwth-aachen.de",
    license="MIT",
    packages=[
        "OpenDHW",
        "OpenDHW.Data",
        "OpenDHW.utils"
    ],
    package_data={
        "OpenDHW.Data": ["*.json"]
    },
    install_requires=["scipy", "pathlib", "pandas", "numpy", "matplotlib", "seaborn", "holidays"],
)
