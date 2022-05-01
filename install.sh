pip install -q condacolab
python -c "import condacolab; condacolab.install()"
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
python -c "import torch; print(torch.version.cuda)"
pip install pydicom
pip install vcam
pip install pyyaml==5.1
TORCH_VERSION=$(python thistorch.py)
CUDA_VERSION=$(python thiscuda.py)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VERSION.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_VERSION.html
pip install torch-geometric