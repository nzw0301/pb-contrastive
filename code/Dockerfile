FROM ubuntu:18.04

USER root

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install \
        build-essential \
        bzip2 \
        ca-certificates \
        cmake \
        curl \
        emacs \
        git \
        libjpeg-dev \
        libpng-dev \
        locales \
        sudo \
        unzip \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# install pyenv
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

# install miniconda3 and packages
RUN pyenv install miniconda3-latest
RUN pyenv global miniconda3-latest

# install dependencies
ADD requirements.txt /

RUN conda install --yes --file requirements.txt && \
    conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch  && \
    conda clean -tipsy
