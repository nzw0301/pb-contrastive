# Codes for [PAC-Bayesian Contrastive Unsupervised Representation Learning](https://arxiv.org/abs/1910.04464)

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

Run codes on [`CNN-README.md`](./code/CNN-README.md) and [`MLP-README.md`](./code/MLP-README.md) under [`code`](./code) dir.
Then run [`create-tables.ipynb`](./code/create-tables.ipynb) to create tables in the main paper.

Option:
Run codes on [`non-iid-README.md`](code/non-iid-README.md), then run [`create-tables-in-Appendix.ipynb`](./code/create-tables-in-Appendix.ipynb) to create tables in the appendix.

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

## Related resources

- [Presentation slides](https://nzw0301.github.io/assets/pdf/uai2020.pdf)
- [Presentation on Youtube](https://www.youtube.com/watch?v=s-PrWBoakw0)

## Reference

```
@inproceedings{NGG2020,
    title = {PAC-Bayesian Contrastive Unsupervised Representation Learning},
    author = {Kento Nozawa, Pascal Germain, Benjamin Guedj},
    year = {2020},
    booktitle = {UAI},
    pages = {21--30}
}
```
