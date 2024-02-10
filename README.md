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
### Discription of files

#### deepHGT.h5 
the weight of DeepHGT.

#### deepHGT_train.py 
Training DeepHGT using file train_valid_test_data.txt for HGT insertion sites recognition.

#### deepHGT_eval.py 
Evaluating DeepHGT using file independent_test_data.txt.

#### deepHGT_pred.py 
Predicting HGT site using DeepHGT for DNA sequences in a given txt file.

### Download data sets
The two data sets including train_valid_test_data.txt and independent_test_data.txt should be downloaded from google drive.

#### train_valid_test_data.txt 
1,556,694 sequences for training (90%), validation (10%), and testing (10%) DeepHGT. Half of the data set are near HGT insertion sites and the remaining half are random sequences extracted from the reference genomes. This file could be downloaded from the shared link [https://drive.google.com/file/d/1Ja2w5TjfCcQyuMl2N73jwQP_-97n3aEC/view?usp=sharing](https://drive.google.com/file/d/1dfn67bM7Sh-c7s5creX48ic8M0-b5ymk/view?usp=drive_link)

#### independent_test_data.txt
689,332 sequences for testing DeepHGT. Half of the data set are near HGT insertion sites and the remaining half are random sequences extracted from the reference genomes. This file could be downloaded from the shared link [https://drive.google.com/file/d/18K6Xx2mUb4yCkzA5-xW2CWHoWwBknhy4/view?usp=sharing](https://drive.google.com/file/d/1ahjTIEVLxvnOzZVfLWJBAn8xh8URwsa2/view?usp=drive_link)

## DeepHGT usage
### Train DeepHGT.
```
usage: python deepHGT_train.py [options]
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
usage: python deepHGT_eval.py [options]
```
#### Required arguments  
  ```
  -i STR  Path to file independent_test_data.txt
  ```

### Predicting HGT sites.
```
usage: python deepHGT_pred.py [options]
```
#### Required arguments  
  ```
  -i FILE  a txt file containing DNA sequences. The length of DNA sequence in each row is 100bp.  
  -w FILE  weight of DeepHGT ["DeepHGT.h5"]
  -o FILE  prediction result
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
### Predict HGT sites
```
python deepHGT_pred.py -i seq.txt -w DeepHGT.h5 -o prediction.txt
```
