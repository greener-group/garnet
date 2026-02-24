# Training the Garnet force field

## Setup

Julia is required, with Julia 1.11 being used in our case.
In the Julia REPL, press `]` to enter the Pkg mode and enter:
```
add Molly@0.23.1 Flux@0.16.5 GraphNeuralNetworks@1.0.0 Zygote@0.7.10 Enzyme@0.13.104 HDF5 Polynomials BSON ChainRulesCore Chemfiles CairoMakie SQLite DataFrames TimerOutputs
```
This combination of versions worked in our case, other versions may not work.
An exact Manifest.toml is available in this directory.

From this directory, run:
```bash
# Download SPICE data (38 GB)
wget -O SPICE-2.0.1.hdf5 https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5?download=1

# Download GEMS data (47 MB)
wget -O crambin.db https://zenodo.org/records/10720941/files/crambin.db?download=1

# Download Espaloma data (170 MB)
wget -O RNA-DIVERSE-OPENFF-DEFAULT.hdf5 https://zenodo.org/records/8148817/files/RNA-DIVERSE-OPENFF-DEFAULT.hdf5?download=1
wget -O RNA-NUCLEOSIDE-OPENFF-DEFAULT.hdf5 https://zenodo.org/records/8148817/files/RNA-NUCLEOSIDE-OPENFF-DEFAULT.hdf5?download=1
wget -O RNA-TRINUCLEOTIDE-OPENFF-DEFAULT.hdf5 https://zenodo.org/records/8148817/files/RNA-TRINUCLEOTIDE-OPENFF-DEFAULT.hdf5?download=1

# Download MACE-OFF data (1.6 GB)
cd data_maceoff
wget -O train_large_neut_no_bad_clean.tar.gz https://www.repository.cam.ac.uk/bitstreams/b185b5ab-91cf-489a-9302-63bfac42824a/download
wget -O test_large_neut_all.tar.gz https://www.repository.cam.ac.uk/bitstreams/cb8351dd-f09c-413f-921c-67a702a7f0c5/download
tar -zxvf train_large_neut_no_bad_clean.tar.gz
tar -zxvf test_large_neut_all.tar.gz
mkdir water
julia extract_water.jl
rm train_large_neut_no_bad_clean.xyz test_large_neut_all.xyz train_large_neut_no_bad_clean.tar.gz test_large_neut_all.tar.gz
cd ..

# Install simulation environment
conda create -n openmm python=3.11
conda activate openmm
conda install -c conda-forge openmm libstdcxx-ng llvm-openmp openff-toolkit openmmforcefields mdanalysis rdkit
conda deactivate

# Install nmrgnn
conda create --name nmr python=3.9
conda activate nmr
conda install cudatoolkit cudnn=8
pip install tensorflow-gpu==2.6.0 click==8.0.1 pandas==1.3.0 tqdm==4.61.2 nmrgnn-data==0.7 keras-tuner==1.0.2 scipy==1.7.0 keras==2.6.0 protobuf==3.20.0 nmrgnn
pip install MDAnalysis==2.7.0 numpy==1.22.4
conda deactivate

# Run reference simulations
# Change non-bonded 1-4 scaling (NonbondedForce line) in
#   /path/to/miniconda3/envs/openmm/lib/python3.11/site-packages/openmmforcefields/ffxml/amber/gaff/ffxml/gaff-2.11.xml
#   to match TIP3P force field file value (0.833333)
# Change chem_shift_path in sim_training.py to point to the nmr conda environment
conda activate openmm
python sim_training.py condensed_data/trajs_ref
conda deactivate

# Re-combine files split to work round GitHub file size limits
cd features
cat features_split_1.tsv features_split_2.tsv features_split_3.tsv features_split_4.tsv > features.tsv
cd ..

# Change submit_training_sims in train.jl to launch an appropriate job on your system
# This will involve calling the above openmm environment
```

## Training

The model trains multi-threaded on CPU with training having high memory requirements (we used 754 GB).
Training simulations run after a few epochs are spawned as extra jobs to be run on GPU.
This means the script is best run on the CPU node of a cluster with access to GPU nodes.
However, it can be run on one system in which case CPU training and GPU simulations would be run serially.

To train in a new directory `trained_model`, run:
```bash
julia -t 32 train.jl trained_model
```
The Enzyme warning `TODO forward zero-set of memorycopy used memset rather than runtime type` can be ignored.
Sometimes the training simulations, first submitted after epoch 4, give NaNs and do not complete.
This can be checked in the `trained_model/training_sims` directory.
In this case the training job will wait forever and should be cancelled.

Training takes around 5 days, with the model after 12 epochs treated as our final model.
As described in the paper, the final model was selected from a number of repeats.

Getting training working is not trivial; if you run into problems, do open an issue.
