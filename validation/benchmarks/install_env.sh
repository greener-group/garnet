# 1. Create the base environment
conda create -n espl python=3.11 numpy=1.26.4 -y
conda activate espl

# 2. Install ALL conda-forge dependencies first.
# We explicitly include libstdcxx-ng to ensure DGL and PyG have the correct C++ backend.
conda install -c conda-forge ase openmm openff-toolkit openmmforcefields openff-qcsubmit libstdcxx-ng -y

# 3. Install PyTorch FIRST among the pip packages.
# This establishes the C++ API for the subsequent ML libraries.
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

# 4. Install DGL, pointing to the specific wheel repository.
pip install dgl==2.5.0 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

# 5. Install PyTorch Geometric and its C++ extensions.
# These must match the PyTorch 2.3.0 and CUDA 12.1 configuration established in step 3.
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# 6. Install the remaining pure-Python or non-conflicting pip packages.
pip install pandas matplotlib qcportal torchdata==0.9.0 ipython pydantic==2.11.9 h5py

# 7. Install the Espaloma force field.
python Models/Espaloma/setup.py install
