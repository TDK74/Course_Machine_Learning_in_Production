import lab_utils
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split


## ------------------------------------------------------ ##
BASE_DIR = './E1'

data_dir, model_dir, vocab_dir = lab_utils.set_experiment_dirs(BASE_DIR)

print(f'base directory: {BASE_DIR}\n\n'
    f'data: {data_dir}\n'
    f'model: {model_dir}\n'
    f'vocab: {vocab_dir}\n')

#print(f'base directory: {BASE_DIR}\n\ndata: {data_dir}\nmodel: {model_dir}\nvocab: {vocab_dir}\n')

## ------------------------------------------------------ ##
pd.set_option('display.max_colwidth', None)

train_df = pd.read_csv(f'{data_dir}/train_data.csv')
test_df = pd.read_csv(f'{data_dir}/test_data.csv')

train_df[ : 10]

## ------------------------------------------------------ ##


## ------------------------------------------------------ ##


## ------------------------------------------------------ ##


## ------------------------------------------------------ ##


## ------------------------------------------------------ ##


## ------------------------------------------------------ ##


## ------------------------------------------------------ ##


## ------------------------------------------------------ ##


## ------------------------------------------------------ ##


## ------------------------------------------------------ ##


## ------------------------------------------------------ ##


## ------------------------------------------------------ ##
