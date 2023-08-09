from setuptools import setup

setup(
    name="arcle",
    version="0.0.1",
    install_requires=["gymnasium==0.29.0", "pygame>=2.0.0", "numpy>=1.25.0"],
    data_files=[('arcs/ARC/data/training',['*.json']),('arcs/ARC/data/evaluation',['*.json'])]
)