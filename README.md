# DeepHGT
This project contains codes and data for paper "Deep learning for HGT insertion sites recognition".
## Table of Contents
1. [Installation](#installation)
2. [DeepHGT usage](#DeepHGT-usage)
3. [Example workflow](#example-workflow)
## Installation
### Requirements
- Softwares
  - Python(3.5+)
- Python packages(3.5+)
  - keras
  - sklearn

### Install
Download and install
```
git clone --recursive https://github.com/lichen2018/DeepHGT.git
cd DeepHGT
wget -N https://media.githubusercontent.com/media/lichen2018/DeepHGT/master/train_validate_test_data.txt
wget -N https://media.githubusercontent.com/media/lichen2018/DeepHGT/master/independent_test_data.txt
wget -N https://media.githubusercontent.com/media/lichen2018/DeepHGT/master/independent_test_label.txt
```

## Discription of files

#### deepHGT.h5 
the weight of DeepHGT.

#### train_valid_test_data.txt 
1,556,694 sequences for training (90%), validation (10%), and testing (10%) DeepHGT. Half of the data set are near HGT insertion sites and the remaining half are random sequences extracted from the reference genomes.

#### independent_test_data.txt
689,332 sequences for testing DeepHGT. Half of the data set are near HGT insertion sites and the remaining half are random sequences extracted from the reference genomes.

#### independent_test_label.txt
label information for the independent test data.

#### deepHGT_train.py 
Training DeepHGT using file train_valid_test_data.txt for HGT insertion sites recognition.

#### deepHGT_eval.py 
Evaluating DeepHGT using file independent_test_data.txt.

## DeepHGT usage
### Train DeepHGT.
```
usage: python DeepHGT/deepHGT_train.py [options]
```
#### Option arguments
  ```
  -o FILE  Image of training process ["training.pdf"]
  -w FILE  weight of DeepHGT ["weight.h5"]
  ```
### Evaluate DeepHGT.
```
usage: python DeepHGT/deepHGT_eval.py
```

## Example workflow
### Training DeepHGT
```
python DeepHGT/deepHGT_train.py
```
### Evaluate DeepHGT
```
python DeepHGT/deepHGT_eval.py
```
