import os.path as pt
from setuptools import setup, find_packages

PACKAGE_DIR = pt.abspath(pt.join(pt.dirname(__file__)))

packages = find_packages(PACKAGE_DIR)

package_data = {
    package: [
        '*.py',
        '*.txt',
        '*.json',
        '*.npy'
    ]
    for package in packages
}

with open(pt.join(PACKAGE_DIR, 'requirements.txt')) as f:
    dependencies = [l.strip(' \n') for l in f]

setup(
    name="OECL",
    version="0.1",
    description="Understanding Normalization in Contrastive Representation Learning and Out-of-Distribution Detection",
    packages=packages,
    package_data=package_data,
    install_requires=dependencies,
)