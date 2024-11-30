#!/bin/bash
# setup.sh

# Install base requirements
pip install -r requirements.txt

# Install PyG dependencies with correct CUDA version
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.4.0+cu121.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.4.0+cu121.html
