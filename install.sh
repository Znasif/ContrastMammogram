python -c "import torch; print(torch.version.cuda)"
pip install pydicom
pip install vcam
pip install pyyaml==5.1
TORCH_VERSION=$(python thistorch.py)
CUDA_VERSION=$(python thiscuda.py)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install torch-scatter -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VERSION.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VERSION.html
pip install torch-geometric