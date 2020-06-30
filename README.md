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
  - numpy

### Install
Download and install
```
git clone --recursive https://github.com/lichen2018/DeepHGT.git
cd DeepHGT
```
### Download data sets
The two data sets including train_valid_test_data.txt and independent_test_data.txt should be downloaded from google drive.
#### train_valid_test_data.txt 
1,556,694 sequences for training (90%), validation (10%), and testing (10%) DeepHGT. Half of the data set are near HGT insertion sites and the remaining half are random sequences extracted from the reference genomes. The shared link is https://drive.google.com/file/d/1Ja2w5TjfCcQyuMl2N73jwQP_-97n3aEC/view?usp=sharing

#### independent_test_data.txt
689,332 sequences for testing DeepHGT. Half of the data set are near HGT insertion sites and the remaining half are random sequences extracted from the reference genomes. The shared link is https://drive.google.com/file/d/18K6Xx2mUb4yCkzA5-xW2CWHoWwBknhy4/view?usp=sharing

## Discription of files

#### deepHGT.h5 
the weight of DeepHGT.

#### deepHGT_train.py 
Training DeepHGT using file train_valid_test_data.txt for HGT insertion sites recognition.

#### deepHGT_eval.py 
Evaluating DeepHGT using file independent_test_data.txt.

## DeepHGT usage
### Train DeepHGT.
```
usage: python DeepHGT/deepHGT_train.py [options]
```
#### Required arguments  
  ```
  -i STR  Path to file train_valid_test_data.txt
  ```
#### Option arguments
  ```
  -o FILE  Image of training process ["training.pdf"]
  -w FILE  weight of DeepHGT ["weight.h5"]
  ```
### Evaluate DeepHGT.
```
usage: python DeepHGT/deepHGT_eval.py [options]
```
#### Required arguments  
  ```
  -i STR  Path to file independent_test_data.txt
  ```

## Example workflow
### Training DeepHGT
```
python deepHGT_train.py -i Path to train_valid_test_data.txt
```
### Evaluate DeepHGT
```
python deepHGT_eval.py -i Path to independent_test_data.txt
```
