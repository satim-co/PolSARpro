<a id="readme-top"></a>
# **PolSARpro**

![Stars](https://img.shields.io/github/stars/satim-co/PolSARpro?style=flat-square&label=Stars)
![Contributors](https://img.shields.io/github/contributors/satim-co/PolSARpro.svg?label=Contributors)
![Issues](https://img.shields.io/github/issues/satim-co/PolSARpro?label=Issues)
![Languages](https://img.shields.io/github/languages/top/satim-co/PolSARpro)

<img align="left" src="https://avatars.githubusercontent.com/u/104204037?s=200&v=4">

<br>

[satim-co/PolSARpro](https://github.com/satim-co/PolSARpro)

<br clear="left"/>

## **Table of contents**
* [Introduction](#introduction)
* [System Requirements](#system-requirements)
* [Installation](#installation)
* [Features](#features)
* [Simple benchmark of processing times for py and c versions](#simple-benchmark-of-processing-times-for-py-and-c-versions)

## **Introduction**
PolSarPro Python version is an advanced open-source tool for processing polarimetric SAR (Synthetic Aperture Radar) data. This project is based on the PolSARpro Bio, version 6.0 (currently implemented in C) described on https://step.esa.int/main/toolboxes/polsarpro-v6-0-biomass-edition-toolbox/ and
available for free on https://ietr-lab.univ-rennes1.fr/polsarpro-bio/,  but re-implemented in Python, enabling better integration with modern scientific libraries and providing flexibility and performance.
PolSARPro.Py is a python library of selected polarimetric functionalities , extracted from the educational scientific PolSARpro Bio toolbox, that performs comprehensive processing without the need for additional software.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## **System Requirements**
To run PolSARPro.py effectively, the system must meet the following prerequisites:
- Python Version: Python 3.11 or higher.
Required Libraries: The software depends on specific versions of various libraries. These are listed in the requirements.txt file, including:
- numba==0.60.0
- numpy==2.0.1
- scipy==1.14.0
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## **Installation**
To install PolSARpro.py Software, user has to:
Download the PolSARPro Python Library. Obtain the latest version of the PolSARPro.py software from the official source or repository:
`https://github.com/satim-co/PolSARpro`

Clone the repo
   ```sh
   git clone https://github.com/satim-co/PolSARpro
   ```
Obtain the requirements.txt File. This file lists all the necessary Python libraries and their respective versions. Ensure you have this file, which should include the following:
- numba==0.60.0
- numpy==2.0.1
- scipy==1.14.0

Follow these steps to install the PolSARPro.py software:
1. Install Required Libraries: Navigate to the directory containing the requirements.txt file in your terminal or command prompt. Run the following command to install the required Python libraries:
   ```sh
   pip install -r requirements.txt
   ```
      This command will automatically download and install the versions of the libraries specified in the requirements.txt file.
2. Verify Installation. After installation, you can verify if the libraries are installed correctly by using pip list or trying to import them in a Python interpreter.
Post-Installation Steps. Follow any additional post-installation instructions specific to PolSARPro.py to complete the setup.

It is strictly recommended to not change, extract, move or modify any component  included in the PolSARpro.py Software directory and / or change its structure.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## **Features**

- **Polarimetric decompositions**

| Module | Description |
| --- | --- |
| data_process_sngl/h_a_alpha_decomposition.py |  H/A/Alpha Decomposition |
| data_process_sngl/freeman_decomposition.py | Freeman 3 Component Decomposition |
| data_process_sngl/cloude_decomposition.py | Cloude Decomposition |
| data_process_sngl/yamaguchi_3components_decomposition.py | Yamaguchi 3 Component Decomposition |
| data_process_sngl/yamaguchi_4components_decomposition.py | Yamaguchi 4 Component Decomposition |
| data_process_sngl/freeman_2components_decomposition.py | Freeman 2 Component Decomposition |
| data_process_sngl/arii_anned_3components_decomposition.py | Arri 3 Component NNED Decomposition |
| data_process_sngl/arii_nned_3components_decomposition.py | Arri 3 Component ANNED Decomposition |
| data_process_sngl/vanzyl92_3components_decomposition.py | Van Zyl (1992) 3 Component Decomposition |

- **Polarimetric segmentations**

| Module | Description |
| --- | --- |
| data_process_sngl/id_class_gen.py | Basic Scattering Mechanism Identification |
| data_process_sngl/wishart_h_a_alpha_classifier.py | H/A/Alpha Classifiaction |
| data_process_sngl/wishart_supervised_classifier.py | Wishart Supervised Classifier |

- **Polarimetric data processing**

| Module | Description |
| --- | --- |
| data_process_sngl/OPCE.py | Optimisation of the Polarimetric Contrast Enhancement |

- **Polarimetric speckle filter**

| Module | Description |
| --- | --- |
| speckle_filter/boxcar_filter.py | Box Car Filter |
| speckle_filter/lee_refined_filter.py | Refined Lee Filter |

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## **Simple benchmark of processing times for py and c versions**

- System -  VBOX virtual machine:

        * Description:    Ubuntu 22.04.3 LTS
        * Release:        22.04
        * Architecture:   x86_64
        * CPU op-mode(s): 32-bit, 64-bit
        * Address sizes:  40 bits physical, 48 bits virtual
        * CPU(s):         4
        * Vendor ID:      GenuineIntel
        * Model name:     Intel(R) Xeon(R) CPU E5645  @ 2.40GHz
        * Codename:       jammy

```bash
+=====+======================================+=====================================+===========+==========+
| Np. | MODULE                               | INFO                                | TIME 'py' | TIME 'c' |
+=====+======================================+=====================================+===========+==========+
| 1   | arii_anned_3components_decomposition | long time processing about 12:00:00 | 0:00:00   | 0:00:00  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 2   | arii_nned_3components_decomposition  |                                     | 0:02:01   | 0:00:24  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 3   | freeman_2components_decomposition    |                                     | 0:01:47   | 0:00:22  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 4   | id_class_gen                         |                                     | 0:06:01   | 0:00:19  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 5   | OPCE                                 |                                     | 0:57:54   | 1:16:58  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 6   | vanzyl92_3components_decomposition   |                                     | 0:02:09   | 0:00:23  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 7   | wishart_supervised_classifier        |                                     | 0:04:32   | 0:00:25  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 8   | wishart_h_a_alpha_classifier         |                                     | 0:20:32   | 0:04:35  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 9   | lee_refined_filter                   |                                     | 0:01:23   | 0:01:14  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 10  | boxcar_filter                        |                                     | 0:01:07   | 0:00:21  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 11  | cloude_decomposition                 |                                     | 0:02:25   | 0:00:53  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 12  | freeman_decomposition                |                                     | 0:01:39   | 0:01:03  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 13  | h_a_alpha_decomposition              |                                     | 0:02:57   | 0:01:25  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 14  | yamaguchi_3components_decomposition  |                                     | 0:01:32   | 0:00:25  |
+-----+--------------------------------------+-------------------------------------+-----------+----------+
| 15  | yamaguchi_4components_decomposition  |                                     | 0:01:41   | 0:00:27  |
+==============================================================================================+==========+
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
