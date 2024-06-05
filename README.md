# PolSarPro

![Stars](https://img.shields.io/github/stars/satim-co/PolSARpro?style=flat-square&label=Stars)
![Contributors](https://img.shields.io/github/contributors/satim-co/PolSARpro.svg?label=Contributors)
![Issues](https://img.shields.io/github/issues/satim-co/PolSARpro?label=Issues)
![Languages](https://img.shields.io/github/languages/top/satim-co/PolSARpro)

## Introduction
PolSarPro Python version is an advanced open-source tool for processing polarimetric SAR (Synthetic Aperture Radar) data. This project is based on the PolSARpro Bio, version 6.0 (currently implemented in C) described on https://step.esa.int/main/toolboxes/polsarpro-v6-0-biomass-edition-toolbox/ and
available for free on https://ietr-lab.univ-rennes1.fr/polsarpro-bio/,  but re-implemented in Python, enabling better integration with modern scientific libraries and providing flexibility and performance.
PolSARPro.Py is a python library of selected polarimetric functionalities , extracted from the educational scientific PolSARpro Bio toolbox, that performs comprehensive processing without the need for additional software.

## System Requirements
To run PolSARPro.py effectively, the system must meet the following prerequisites:
- Python Version: Python 3.11 or higher.
Required Libraries: The software depends on specific versions of various libraries. These are listed in the requirements.txt file, including:
- numpy==1.25.0
- numba==0.58.0

## Installation
To install PolSARpro.py Software, user has to:
Download the PolSARPro Python Library. Obtain the latest version of the PolSARPro.py software from the official source or repository:
`https://github.com/satim-co/PolSARpro`
Obtain the requirements.txt File. This file lists all the necessary Python libraries and their respective versions. Ensure you have this file, which should include the following:
- numpy==1.25.0
- numba==0.58.0


Follow these steps to install the PolSARPro.py software:
1. Install Required Libraries: Navigate to the directory containing the requirements.txt file in your terminal or command prompt. Run the following command to install the required Python libraries:
`pip install -r requirements.txt`
      This command will automatically download and install the versions of the libraries specified in the requirements.txt file.
2. Verify Installation. After installation, you can verify if the libraries are installed correctly by using pip list or trying to import them in a Python interpreter.
Post-Installation Steps. Follow any additional post-installation instructions specific to PolSARPro.py to complete the setup.

It is strictly recommended to not change, extract, move or modify any component  included in the PolSARpro.pyy Software directory and / or change its structure.
