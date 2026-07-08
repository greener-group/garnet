# Small molecule condensed phase properties

## Background

This repository contains scripts and data to benchmark the Garnet force field against experimental measurements of the density and enthalpy of vapourisation of small molecules. Addtionally, the scripts included can be used to compare Garnet predictions of these properties to OpenFF-2.2.1 and GAFF predictions. 

## Directory content

Experimental density and enthalpy of vapourisation data [1,2] can be found in `condensed_data/exp_data`.

Using the experimental densities, simulation starting structures can be generated with RDKit and PACKMOL by running the script `setup.py`. Running this script will generate data in the remaining folders in `condensed_data/`, including simulations starting structures and PACKMOL input and log files.

Once starting structures have been generated, simulations can be run with `sim.py`, which takes force field ("gaff", "openff" or "garnet") an as input argument. 

Finally, the Jupyter Notebook `analysis.ipynb` contains code to analyse the simulation output and compare simulation results to the experimental data. 

Details about the experimental data and simulation and analysis methods can be found in our paper [3].

## References

1. Wang and Hou. Application of Molecular Dynamics Simulations in Molecular Property Prediction. 1. Density and Heat of Vaporization. J. Chem. Theory Comput. 2011, 7, 7, 2151–2165. DOI: https://doi.org/10.1021/ct200142z

2. Acree and Chickos. Phase Transition Enthalpy Measurements of Organic and Organometallic Compounds. NIST Chemistry WebBook, NIST Standard Reference Database Number 69. DOI: https://doi.org/10.18434/T4D303

3. Blanco-González, Schulze, Rovers and Greener. Training a force field for proteins and small molecules from scratch. arXiv. 2026. DOI: https://doi.org/10.48550/arXiv.2603.16770