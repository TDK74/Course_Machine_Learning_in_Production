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
train_df.topic.value_counts(normalize = True).sort_index().mul(100).round(1).astype(str) + '%'

## ------------------------------------------------------ ##
test_df.topic.value_counts(normalize = True).sort_index().mul(100).round(1).astype(str) + '%'

## ------------------------------------------------------ ##
train_df = pd.read_csv(f'{data_dir}/train_data.csv')
test_df = pd.read_csv(f'{data_dir}/test_data.csv')

combined_df = pd.concat([train_df, test_df], ignore_index = True)

train_df, temp_df = train_test_split(combined_df, test_size = 0.4, stratify = combined_df['topic'],
                                    random_state = 42)

dev_df, test_df = train_test_split(temp_df, test_size = 0.5, stratify = temp_df['topic'],
                                    random_state = 42)

## ------------------------------------------------------ ##
# or in this way:
train_df, test_df = train_test_split(combined_df, test_size = 0.2, stratify = combined_df['topic'])

train_df, dev_df = train_test_split(train_df, test_size = 0.25, stratify = train_df['topic'])

## ------------------------------------------------------ ##
train_df.topic.value_counts(normalize = True).sort_index().mul(100).round(1).astype(str) + '%'

## ------------------------------------------------------ ##
dev_df.topic.value_counts(normalize = True).sort_index().mul(100).round(1).astype(str) + '%'

## ------------------------------------------------------ ##
test_df.topic.value_counts(normalize = True).sort_index().mul(100).round(1).astype(str) + '%'

## ------------------------------------------------------ ##
BASE_DIR = './E2'

data_dir, model_dir, vocab_dir = lab_utils.set_experiment_dirs(BASE_DIR)

lab_utils.save_data(train_df, data_dir, 'train_data.csv')
lab_utils.save_data(dev_df, data_dir, 'dev_data.csv')
lab_utils.save_data(test_df, data_dir, 'test_data.csv')

lab_utils.save_labels(topic_lookup, vocab_dir)

## ------------------------------------------------------ ##
BASE_DIR = './E1'

_, model_dir, _ = lab_utils.set_experiment_dirs(BASE_DIR)

model = tf.keras.models.load_model(model_dir)

BASE_DIR = './E2'

MAX_LENGTH = 20
VOCAB_SIZE = 10000

data_dir, model_dir, vocab_dir = lab_utils.set_experiment_dirs(BASE_DIR)

train_df = pd.read_csv(f'{data_dir}/train_data.csv')
dev_df = pd.read_csv(f'{data_dir}/dev_data.csv')
test_df = pd.read_csv(f'{data_dir}/test_data.csv')

title_preprocessor = tf.keras.layers.TextVectorization(max_tokens = VOCAB_SIZE,
                                                        output_sequence_length = MAX_LENGTH)

topic_lookup = tf.keras.layers.StringLookup(vocabulary = f'{vocab_dir}/labels.txt',
                                            num_oov_indices = 0)

## ------------------------------------------------------ ##
train_inputs = train_df['title']

title_preprocessor.adapt(train_inputs)

lab_utils.save_vocab(title_preprocessor, vocab_dir)

## ------------------------------------------------------ ##
NUM_EPOCHS = 5

train_ds = lab_utils.df_to_tfdata(train_df, topic_lookup, title_preprocessor, shuffle = True)
dev_ds = lab_utils.df_to_tfdata(dev_df, topic_lookup, title_preprocessor)
test_ds = lab_utils.df_to_tfdata(test_df, topic_lookup, title_preprocessor)

model = lab_utils.model_reset_weights(model)

model.fit(train_ds, epochs = NUM_EPOCHS, validation_data = dev_ds, verbose = 1)

## ------------------------------------------------------ ##
model.evaluate(test_ds)

## ------------------------------------------------------ ##
model.save(model_dir)

## ------------------------------------------------------ ##
topics = topic_lookup.get_vocabulary()

lab_utils.print_metric_per_topic(dev_df, topics, topic_lookup, title_preprocessor, model)

## ------------------------------------------------------ ##
train_df[train_df.topic == 'BUSINESS']

## ------------------------------------------------------ ##
BASE_DIR = './E3'

data_dir, model_dir, vocab_dir = lab_utils.set_experiment_dirs(BASE_DIR)

combined_df = pd.read_csv(f'./.backup.csv')

train_df, test_df = train_test_split(combined_df, test_size = 0.2, stratify = combined_df['topic'])
train_df, dev_df = train_test_split(train_df, test_size = 0.25, stratify = train_df['topic'])

lab_utils.save_data(train_df, data_dir, 'train_data.csv')
lab_utils.save_data(dev_df, data_dir, 'dev_data.csv')
lab_utils.save_data(test_df, data_dir, 'test_data.csv')

## ------------------------------------------------------ ##
train_inputs = train_df['title']
title_preprocessor.adapt(train_inputs)

lab_utils.save_vocab(title_preprocessor, vocab_dir)
lab_utils.save_labels(topic_lookup, vocab_dir)

## ------------------------------------------------------ ##
NUM_EPOCHS = 5

train_ds = lab_utils.df_to_tfdata(train_df, topic_lookup, title_preprocessor, shuffle = True)
dev_ds = lab_utils.df_to_tfdata(dev_df, topic_lookup, title_preprocessor)
test_ds = lab_utils.df_to_tfdata(test_df, topic_lookup, title_preprocessor)

model = lab_utils.model_reset_weights(model)

model.fit(train_ds, epochs = NUM_EPOCHS, validation_data = dev_ds, verbose = 1)

## ------------------------------------------------------ ##
model.evaluate(test_ds)

model.save(model_dir)

## ------------------------------------------------------ ##
lab_utils.print_metric_per_topic(dev_df, topics, topic_lookup, title_preprocessor, model)

## ------------------------------------------------------ ##
lab_utils.get_errors(model, dev_df, title_preprocessor, topic_lookup, 'NATION')

## ------------------------------------------------------ ##
