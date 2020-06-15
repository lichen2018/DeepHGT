import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re
import random
from sklearn import preprocessing
import numpy as np
from random import shuffle
import pandas as pd
import os
from glob import glob
import fnmatch
import argparse
from scipy import misc
from os.path import join
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, concatenate, SpatialDropout1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Embedding, \
    SeparableConv1D, Add, BatchNormalization, Activation, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, Concatenate, Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D, add
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from keras.models import Model, Sequential
from keras.applications.nasnet import  preprocess_input
from keras.optimizers import Adam, SGD
import math


def loadTrainSeq(file_name):
    fi = open(file_name, "r")
    sample_seq_list = []
    label_list = []
    sample_list = []
    while 1:
        buf = fi.readline()
        buf = buf.strip('\n')
        if not buf:
            break
        buf = re.split(r'[:;,\s]\s*', buf)
        sample_list.append(buf)
    shuffle(sample_list)
    for buf in sample_list:
        sample_seq_list.append(encodeSequence(buf[-2]))
        label_list.append(int(buf[-1]))
    sample_x = np.array(sample_seq_list)
    return sample_x, label_list


def loadSeq(file_name):
    fi = open(file_name, "r")
    sample_seq_list = []
    sample_list = []
    while 1:
        buf = fi.readline()
        buf = buf.strip('\n')
        if not buf:
            break
        buf = re.split(r'[:;,\s]\s*', buf)
        sample_list.append(buf)
    for buf in sample_list:
        sample_seq_list.append(encodeSequence(buf[-2]))
    sample_x = np.array(sample_seq_list)
    return sample_x


def loadLabel(file_name):
    fi = open(file_name, "r")
    label_list = []
    while 1:
        buf = fi.readline()
        buf = buf.strip('\n')
        if not buf:
            break
        buf = re.split(r'[:;,\s]\s*', buf)
        label_list.append(int(buf))
    sample_y = np.array(label_list)
    return sample_y


def random_crop_sequence(seq_batch, max_crop_region, train_length):
    """random crop sequence.
    Args:
    seq: a [batch_size, sequence_length] sequence to shift
    max_crop_region: the maximum amount to shift (tf.int32 or int)
    train_length: length of training sequence
    """
    start_pos = random.randint(0, max_crop_region)
    output = seq_batch[:,start_pos:start_pos+train_length, :]
    return output


def encodeSequence(seq):
    seq_code = np.zeros((len(seq), 4))
    for i in range(len(seq)):
        nt = seq[i]
        if nt == 'A':
            seq_code[i, 0] = 1
        elif nt == 'C':
            seq_code[i, 1] = 1
        elif nt == 'G':
            seq_code[i, 2] = 1
        elif nt == 'T':
            seq_code[i, 3] = 1  
        else:
            ni = random.randint(0,3)
            seq_code[i, ni] = 1
    return seq_code


def chunker(seq, size):
    indices = np.arange(len(seq))
    return (indices[pos:pos + size] for pos in range(0, len(seq), size))


def data_gen(x, y, batch_size, aug_flag):
    while True:
        for batch in chunker(x, batch_size):
            batch.sort()
            X = x[batch]
            Y = y[batch]
            X = [x[:train_length] for x in X]
            if aug_flag is True:
                X = [random_snp(x) for x in X]
            X = [preprocess_input(x) for x in X]
            yield np.array(X), np.array(Y)

def random_snp(seq):
    nonzero_indices = np.nonzero(seq)
    for i in range(random_snp_count):
        snp_pos = random.randint(0, len(nonzero_indices[0])-1)
        new_snp = (nonzero_indices[1][snp_pos]+random.randint(0,3))%4
        seq[snp_pos, nonzero_indices[1][snp_pos]] = 0
        seq[snp_pos, new_snp] = 1
    return seq

def residual_block(x):
    shortcut = x

    x = Conv1D(filters=128,  kernel_size=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv1D(filters=128, kernel_size=4, padding='same')(x)
    x = BatchNormalization()(x)

    x = add([shortcut, x])
    x = LeakyReLU()(x)
    return x


def convolutional_block(x):
    x = Conv1D(filters=128,  kernel_size=4, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def get_classification_model(train_seq_length):
    i = Input((train_seq_length, 4))
    x = Conv1D(filters=128, kernel_size=4)(i)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = residual_block(x)
    x = Dropout(0.1)(x)    
    x = residual_block(x)
    x = Dropout(0.1)(x)
    x = residual_block(x)
    x = Dropout(0.25)(x)
    x = residual_block(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(i, x)
    model.compile(loss=binary_crossentropy, optimizer=SGD(0.001), 
                metrics=['acc'])
    model.summary()
    return model



def main():
    batch_size = 120
    max_shift_amount = 10
    train_length = 100
    random_snp_count = 10
    x_file_name = 'independent_test_data.txt'
    sample_x = loadSeq(x_file_name)
    y_file_name = 'independent_test_label.txt'
    sample_y = loadLabel(y_file_name)
    model = get_classification_model(train_length)
    h5_path = "deepHGT.h5"
    model.load_weights(h5_path)
    tmp_y = []
    for batch in chunker(sample_x, batch_size):
        batch.sort()
        X = sample_x[batch]
        X = [x[:train_length] for x in X]
        X = [preprocess_input(x) for x in X]
        y_pred_keras = model.predict(np.array(X)).ravel()
        tmp_y.append(y_pred_keras)
    pred_y = []
    fo=open('prediction.txt', 'w')
    for batch in tmp_y:
        for i in range(batch.shape[0]):
            fo.write(str(batch[i])+'\n')
            pred_y.append(batch[i])
    fo.close()
    fpr, tpr, thresholds = metrics.roc_curve(sample_y, np.array(pred_y))
    auc = round(metrics.auc(fpr, tpr),3)
    precision, recall, _ = precision_recall_curve(sample_y, np.array(pred_y))
    ap = round(average_precision_score(precision, recall), 3)
    fo=open('auc_ap.txt', 'w')
    fo.write("AUC:"+str(auc)+'\n')
    fo.write("AP:"+str(ap)+'\n')
    fo.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DeepHGT.", add_help=False, usage="%(prog)s [-h]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument("-h", "--help", action="help")
    args = vars(parser.parse_args())
    sys.exit(main())
