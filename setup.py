from setuptools import setup

setup(
    name="OpenDHW",
    version="0.1",
    description="Tool for Energy Analysis and Simulation for " "Efficient Retrofit ",
    url="https://github.com/RWTH-EBC/OpenDHW",
    author="RWTH Aachen University, E.ON Energy Research Center, "
    "Institute of Energy Efficient Buildings and Indoor Climate",
    author_email="ebc-teaser@eonerc.rwth-aachen.de",
    license="MIT",
    packages=[
        "OpenDHW",
        "OpenDHW.utils"
    ],
    install_requires=["scipy", "pathlib", "pandas", "numpy", "matplotlib", "seaborn"],
)
