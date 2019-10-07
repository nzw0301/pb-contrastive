## Experimental environment

- Ubuntu 18.04.1 LTS
- Cuda: `10.2`
- Cudnn: `7.6.3`
- `conda`

---


```bash
git clone git@github.com:pb-contrastive.git
cd pb-contrastive
```

### Optional: Install `miniconda3-latest` via `pyenv`

```bash
# cd pb-contrastive

pyenv install miniconda3-latest
pyenv local miniconda3-latest
conda create --name pac-bayes --file requirements.txt -y
pyenv local miniconda3-latest/envs/pac-bayes
```

## Install GPUs supproted `PyTorch`

See also the latest [`PyTorch`](https://github.com/pytorch/pytorch#from-source).

Note that `PyTorch` was `1.2.0`.

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

Run codes on `README.md` and `MLP-README.md` under `code` dir.
Then run `create-tables.ipynb` to create tables in the manuscript.
