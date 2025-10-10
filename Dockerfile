FROM mambaorg/micromamba:debian12-slim
# FROM mambaorg/micromamba

USER root
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN apt-get install -y git python3-pip wget libpq-dev procps gdal-bin
RUN apt-get install -y gcc g++ build-essential 
RUN apt-get install -y unzip
# Extra libs for C-PolSARPro, not needed for the subset of features we consider
# RUN apt-get install -y freeglut3-dev libfreeimage-dev
# RUN apt-get install -y libtk-img iwidgets4 bwidget
# RUN apt-get install -y libglew-dev

# Install C-version of PolSARPro -- zip file needs to be downloaded from
# https://ietr-lab.univ-rennes1.fr/polsarpro-bio/Linux/PolSARpro_v6.0.3_Biomass_Edition_Linux_Installer_20210501.zip
ARG CPSPDIR="/home/c_psp/"
WORKDIR ${CPSPDIR}
COPY PolSARpro_v6.0.4_Biomass_Edition_Linux_Installer_20250122.zip ${CPSPDIR}
RUN unzip ${CPSPDIR}PolSARpro_v6.0.4_Biomass_Edition_Linux_Installer_20250122.zip
# we use custom building scripts
COPY scripts/Compil_PolSARpro_Biomass_Edition_Linux.sh ${CPSPDIR}Soft
COPY scripts/PolSARpro_v6.0.4_Biomass_Edition_Linux_Installation.sh ${CPSPDIR}
RUN chmod +x PolSARpro_v6.0.4_Biomass_Edition_Linux_Installation.sh
RUN ./PolSARpro_v6.0.4_Biomass_Edition_Linux_Installation.sh

# Setup micromamba (lightweight conda clone)
WORKDIR "/tmp/conda_init/"
SHELL [ "/bin/bash", "--login", "-c" ]
RUN micromamba shell init --shell=bash --root-prefix=~/micromamba
RUN source ~/.bashrc
COPY environment.yaml environment.yaml
RUN micromamba create -f  environment.yaml
RUN echo "micromamba activate psp" >> ~/.bashrc
RUN echo "alias conda='micromamba'" >> ~/.bashrc
RUN micromamba activate psp
