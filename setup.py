import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="garnetff",
    version="0.1.0",
    author="Joe G Greener",
    author_email="jgreener@mrc-lmb.cam.ac.uk",
    description="Garnet biomolecular force field",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/greener-group/garnet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    license="MIT",
    keywords="biomolecule force field graph neural network",
    install_requires=["torch", "torch_geometric"],
    include_package_data=True,
)
