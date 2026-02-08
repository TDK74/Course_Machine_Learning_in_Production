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
train_df[['title', 'topic']]

## ------------------------------------------------------ ##
start_index = 30
end_index = 40

train_df[['title']][start_index : end_index]

train_df[['title', 'topic']][start_index : end_index]

## ------------------------------------------------------ ##
model = tf.keras.models.load_model(model_dir)

model.summary()

## ------------------------------------------------------ ##
model.get_compile_config()

## ------------------------------------------------------ ##
topic_lookup = tf.keras.layers.StringLookup(vocabulary = f'{vocab_dir}/labels.txt',
                                            num_oov_indices = 0)

topic_lookup.get_vocabulary()

## ------------------------------------------------------ ##
MAX_LENGTH = 20
VOCAB_SIZE = 10000

title_preprocessor = tf.keras.layers.TextVectorization(max_tokens = VOCAB_SIZE,
                                                        output_sequence_length = MAX_LENGTH)

title_preprocessor.load_assets(vocab_dir)

print(f'vocabulary size: {title_preprocessor.vocabulary_size()}')

sample_title = train_df['title'][10]

print(f"sample text: {sample_title}")

print(f"sample text (preprocessed): {title_preprocessor(sample_title)}")

## ------------------------------------------------------ ##
test_ds = lab_utils.df_to_tfdata(test_df, topic_lookup, title_preprocessor)

model.evaluate(test_ds)

## ------------------------------------------------------ ##
