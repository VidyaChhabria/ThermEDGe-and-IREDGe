# ThermEDGe and IREDGe
Encoder decoder based generative networks for static and transient thermal analysis. This repository contains code for the paper titled "Thermal and IR Drop Analysis Using ConvolutionalEncoder-Decoder Networks". 

[![Standard](https://img.shields.io/badge/python-3.6-blue)](https://commons.wikimedia.org/wiki/File:Blue_Python_3.6_Shield_Badge.svg)
[![Download](https://img.shields.io/badge/Download-here-red)](https://github.com/VidyaChhabria/TherMOS/archive/master.zip)
[![Version](https://img.shields.io/badge/version-0.1-green)](https://github.com/VidyaChhabria/TherMOS/tree/master)
[![AskMe](https://img.shields.io/badge/ask-me-yellow)](https://github.com/VidyaChhabria/TherMOS/issues)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


Fig.7 of the paper is a video which is available below. Please click [here](https://github.com/VidyaChhabria/ThermEDGe-and-IREDGe/raw/master/Fig7.gif) and download and view in any image viewer.


| <img src="Fig7.gif" width=900px> |
|:--:|
| *Video comparing the prediction of transient ThermEDGe against commercial tool-generated temperature contours for Testcase 6: (i) left video shows the time varying power map, (ii) center video shows the commercially-generated temperature contours, and (iii) right video shows ThermEDGe-generated temperature contours|


## How to run:

### Download

`git clone https://github.com/VidyaChhabria/EDGe-Thermal-Analysis.git`

### Install dependencies

- Bare minimum dependencies are the following:
    - python3.6
    - pip-20.1


- Create virtual environment and install the required python packages

```
python3 -m venv EDGe
source EDGe/bin/activate
pip3 install -r requirements.txt
```

### Run the flow
- Default settings for training, ML-hyper parameters, chip sizes, tile-size are mentioned in the
  config.yaml file. Change if required.
- Example to run the flow:
```
python3 src/transient_thermal_model.py -train_data_path ./data/data_set_2/train/Transient_runs -test_data_path ./data/data_set_2/test/Transient_runs -output_plot ./output/.
```
| Argument              	| Comments                                                                             	|
|-----------------------	|--------------------------------------------------------------------------------------	|
| -h, --help            	| Prints out the usage                                                                 	|
| -train_data_path <str>    | Path to the training data runs (required, str)                                        |
| -test_data_path <str>  	| Path to the testing data (str, required)                	                            |
| -output_plot <str>       	| Path to generate the output plots (required,str)                   	                |

Note: 
- The paths here point to the Transient runs data directory as shown in the example
above with the data in the same csv file format and similar naming convention
provided to me:"Transient_runs/Run_%d_contour_data"
- Create two directory trees with the same structure. One for training and one for
testing. 
- Add all the data points for testing into the test/Transient_runs directory

### To do
- Include script for static thermal prediction
- Include script for the other implementation of the model which uses static thermal solution
  as an input to predict transient thermal solution


### How to cite

V. A. Chhabria, V. Ahuja, A. Prabhu, N. Patil, P. Jain, and S. S. Sapatnekar, “Thermal and IR Drop Analysis Us-ing Convolutional Encoder-Decoder Networks,” Proc.  of Asia and South Pacific Design Automation Conference, 2021.

