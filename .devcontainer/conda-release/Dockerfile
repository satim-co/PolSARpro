FROM mambaorg/micromamba:debian12-slim

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/conda-release-env
SHELL [ "/bin/bash", "--login", "-c" ]
RUN micromamba shell init --shell=bash --root-prefix=~/micromamba
COPY environment.yaml environment.yaml

RUN micromamba create -y -f environment.yaml \
    && micromamba clean --all --yes

WORKDIR /polsarpro-dev
COPY README.md CHANGELOG.md LICENSE mkdocs.yml ./

RUN echo "micromamba activate psp" >> ~/.bashrc \
    && echo "alias conda='micromamba'" >> ~/.bashrc \
    && mkdir -p ~/.local/share/jupyter/kernels

ENV PYTHONNOUSERSITE=1

CMD ["/bin/sh", "-c", "while sleep 1000; do :; done"]
