## Experimental environment

- Ubuntu 18.04.1 LTS
- Cuda: `10.2`
- Cudnn: `7.6.3`
- `conda`

---


```bash
git clone git@github.com:nzw0301/pb-contrastive.git
cd pb-contrastive
```

### Optional: Install `miniconda3-latest` via `pyenv`

```bash
# cd pb-contrastive

pyenv install miniconda3-latest
pyenv local miniconda3-latest
conda create --name pac-bayes --file conde/requirements.txt -y
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

Run codes on `CNN-README.md` and `MLP-README.md` under `code` dir.
Then run `create-tables.ipynb` to create tables in the main paper.

Option:
Run codes on `non-iid-README.md`, then run `create-tables-in-Appendix.ipynb` to create tables in the appendix.

### Optional: Install parts of experimental dependencies on CPU via Dockerfile to run the jupyter notebook

We provide a docker environment to run notebooks on your local machine without GPUs.

For `bash/zsh`:

```bash
# cd code
docker build . -t pb-contrastive:latest
docker run -i -p 8888:8888 -v $(pwd):/pb-contrastive/code -w="/pb-contrastive/code" -t pb-contrastive /bin/bash

jupyter notebook --ip=0.0.0.0 --allow-root
```

For `fish`:

```fish
# cd code
docker build . -t pb-contrastive:latest
docker run -i -p 8888:8888 -v (pwd):/pb-contrastive/code -w="/pb-contrastive/code" -t pb-contrastive /bin/bash

jupyter notebook --ip=0.0.0.0 --allow-root
```

---

## BibTeX

```
@inproceedings{NGG2019,
    title = {PAC-Bayesian Contrastive Unsupervised Representation Learning},
    author = {Kento Nozawa, Pascal Germain, Benjamin Guedj},
    year = {2019},
    archivePrefix = {arXiv},
    eprint = {1910.04464},
}
```
