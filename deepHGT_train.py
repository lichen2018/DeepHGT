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
    file_name = 'train_validate_test_data.txt'

    sample_x, sample_y_list = loadTrainSeq(file_name)
    lb = preprocessing.LabelBinarizer()
    lb.fit(sample_y_list)
    sample_y = lb.transform(sample_y_list)

    x_train, x_test, y_train, y_test = train_test_split(sample_x, sample_y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)


    model = get_classification_model(train_length)

    history = model.fit_generator(
        data_gen(x_train, y_train, batch_size, True),
        validation_data=data_gen(x_val, y_val, batch_size, False),
        epochs=600, verbose=1,
        steps_per_epoch=len(x_train) // batch_size,
        validation_steps=len(x_val) // batch_size)

    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.savefig(args["o"])

    to_h5_path = args["w"]
    model.save_weights(to_h5_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate DeepHGT.", add_help=False, usage="%(prog)s [-h] -r genome_dir -id sample_id.txt", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument("-o", type=str, default="training.pdf", help="<str> Image of training process", metavar="\b")
    optional.add_argument("-w", type=str, default="weight.h5", help="<str> weight of DeepHGT.", metavar="\b")
    optional.add_argument("-h", "--help", action="help")
    args = vars(parser.parse_args())
    sys.exit(main())
