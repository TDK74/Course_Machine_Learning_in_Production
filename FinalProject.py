import json
import os
import shutil
import sqlite3
import lab_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
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
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam',
              metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(k = 2)])

model.evaluate(dev_ds)

## ------------------------------------------------------ ##
lab_utils.print_metric_per_topic(dev_df, topics, topic_lookup, title_preprocessor, model)

## ------------------------------------------------------ ##
EMBEDDING_DIM = 24
DENSE_DIM = 24
topic_size = topic_lookup.vocabulary_size()

model = tf.keras.Sequential([tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                                                        input_length = MAX_LENGTH),
                            tf.keras.layers.Dense(DENSE_DIM, activation = 'relu'),
                            tf.keras.layers.Dense(topic_size, activation = 'softmax')])

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam',
            metrics = ['sparse_categorical_accuracy'])

model.summary()

## ------------------------------------------------------ ##
# Uncomment the lines below if you ran this section before and
# want to DELETE all models in the serving directory

# SERVING_DIR = f'{os.getcwd()}/serving'
# os.environ["SERVING_DIR"] = SERVING_DIR
# os.system('find $SERVING_DIR -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} \;')

## ------------------------------------------------------ ##
SERVING_DIR = f'{os.getcwd()}/serving'
os.environ["SERVING_DIR"] = SERVING_DIR

print(f'SERVING_DIR: {SERVING_DIR}')
print(f'os.environ["SERVING_DIR"]: {os.environ["SERVING_DIR"]}')

## ------------------------------------------------------ ##
os.makedirs(f'{SERVING_DIR}/1', exist_ok = True)

shutil.copytree('./E2/model/', f'{SERVING_DIR}/1', dirs_exist_ok = True)

## ------------------------------------------------------ ##
## NOTE: If you're running this notebook outside the DeepLearning.AI platform
## and want to use Docker instead.

# command = ('docker run -p 8501:8501 --mount type=bind,source="${SERVING_DIR}",'
#             'target=/models/newsapp_model -e MODEL_NAME=newsapp_model '
#             '--name=tensorflow-serving -t tensorflow/serving &')

# os.system(command)

## ------------------------------------------------------ ##
command = (f'nohup tensorflow_model_server --rest_api_port=8501 --model_name=newsapp_model '
            f'--model_base_path="{SERVING_DIR}" > ./serving/server.log 2>&1 &')

os.system(command)

## ------------------------------------------------------ ##
json_payload = '{"instances": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}'

command = (f"curl -d '{json_payload}' http://localhost:8501/v1/models/newsapp_model:predict")

os.system(command)

## ------------------------------------------------------ ##
MAX_LENGTH = 20
VOCAB_SIZE = 10000

title_preprocessor = tf.keras.layers.TextVectorization(max_tokens = VOCAB_SIZE,
                                                       output_sequence_length = MAX_LENGTH)

title_preprocessor.load_assets('./E2/vocab')

sample_input = 'Sample title'

sample_input_ds = title_preprocessor(sample_input)

sample_input_ds = tf.expand_dims(sample_input_ds, axis = 0)

data = json.dumps({"instances" : sample_input_ds.numpy().tolist()})

headers = {"content-type" : "application/json"}

json_response = requests.post('http://localhost:8501/v1/models/newsapp_model:predict',
                                data = data, headers = headers)

predictions = json.loads(json_response.text)['predictions']

print(predictions)

## ------------------------------------------------------ ##
BASE_DIR = './E2'

data_dir, model_dir, vocab_dir = lab_utils.set_experiment_dirs(BASE_DIR)

model = tf.keras.models.load_model(model_dir)

title_preprocessor = tf.keras.layers.TextVectorization(max_tokens = VOCAB_SIZE,
                                                       output_sequence_length = MAX_LENGTH)

title_preprocessor.load_assets(vocab_dir)

model_with_preprocessor = tf.keras.Sequential([title_preprocessor, model])

sample_input = "Sample Title"

model_with_preprocessor.predict([sample_input])

## ------------------------------------------------------ ##
model_with_preprocessor.export(f'{SERVING_DIR}/2')

## ------------------------------------------------------ ##
data = json.dumps({"instances" : ["sample title"]})

headers = {"content-type" : "application/json"}

json_response = requests.post('http://localhost:8501/v1/models/newsapp_model:predict',
                                data = data, headers = headers)

predictions = json.loads(json_response.text)['predictions']

print(predictions)

## ------------------------------------------------------ ##
## NOTE: Uncomment and run this cell if you're running on your own device
## and want to use Docker instead.

# command = ('docker stop tensorflow-serving')

# os.system(command)

## ------------------------------------------------------ ##
command = ("kill $(ps aux | grep 'tensorflow_model_server' | awk '{print $2}')")

# or shorter and safer
# command = "pkill -f tensorflow_model_server"

os.system(command)

## ------------------------------------------------------ ##
## NOTE: Copy and uncomment this on a new cell. This will NOT run on the DeepLearning.AI platform.

# command = ('docker run -p 8501:8501 --mount type=bind,source="${SERVING_DIR}",' +
#             'target=/models/newsapp_model --mount type=bind,' +
#             'source="${SERVING_DIR}/models.config-docker",target=/models/models.config ' +
#             '-e MODEL_NAME=newsapp_model --name=tensorflow-serving-models-config -t ' +
#             'tensorflow/serving --model_config_file=/models/models.config ' +
#             '--allow_version_labels_for_unavailable_models=true &')

# os.system(command)

## ------------------------------------------------------ ##
command = (f'nohup tensorflow_model_server --rest_api_port=8501 '
            f'--model_config_file="${SERVING_DIR}/models.config" '
            f'--model_config_file_poll_wait_seconds=10 '
            f'--allow_version_labels_for_unavailable_models=true '
            f'--model_base_path="{SERVING_DIR}" > ./serving/server.log 2>&1 &')

os.system(command)

## ------------------------------------------------------ ##
data = json.dumps({"instances" : [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]})

headers = {"content-type" : "application/json"}
local_addr = 'http://localhost:8501/v1/models/newsapp_model/labels/deprecated:predict'
json_response = requests.post(local_addr, data = data, headers = headers)
predictions = json.loads(json_response.text)['predictions']

print(predictions)

## ------------------------------------------------------ ##
data = json.dumps({"instances" : ["sample title 1", "sample title 2"]})

headers = {"content-type" : "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/newsapp_model/labels/stable:predict',
                                data = data, headers = headers)
predictions = json.loads(json_response.text)['predictions']

print(predictions)

## ------------------------------------------------------ ##
BASE_DIR = './E2'

data_dir, _, vocab_dir = lab_utils.set_experiment_dirs(BASE_DIR)

dev_df = pd.read_csv(f'{data_dir}/dev_data.csv')

topic_lookup = tf.keras.layers.StringLookup(vocabulary = f'{vocab_dir}/labels.txt',
                                            num_oov_indices = 0)

title_df = dev_df['title'][ : 100].reset_index(drop = True)

dev_np = title_df.to_numpy().tolist()

data = json.dumps({"instances" : dev_np})

headers = {"content-type" : "application/json"}

json_response = requests.post('http://localhost:8501/v1/models/newsapp_model/labels/stable:predict',
                                data = data, headers = headers)

predictions = json.loads(json_response.text)['predictions']

## ------------------------------------------------------ ##
pd.options.display.float_format = '{:.2%}'.format

pred_df = pd.DataFrame(predictions, columns = topic_lookup.get_vocabulary())

pred_df = pd.concat([title_df, pred_df], axis = 1)

pred_df

## ------------------------------------------------------ ##
THRESHOLD = 0.6

below_threshold = []

for i, prediction in enumerate(predictions):
    if max(prediction) < THRESHOLD:
        prediction = prediction.copy()
        prediction.insert(0, dev_np[i])
        below_threshold.append(prediction)

columns = topic_lookup.get_vocabulary()
columns.insert(0, 'title')

pd.DataFrame(below_threshold, columns = columns)

## ------------------------------------------------------ ##
con = sqlite3.connect("news_articles.db")
cur = con.cursor()

for row in cur.execute("SELECT id,title FROM news_articles WHERE id < 5"):
    print(row)

## ------------------------------------------------------ ##
VOCAB_SIZE = 10000
MAX_LENGTH = 20

BASE_DIR = './E1'

_, model_dir, vocab_dir = lab_utils.set_experiment_dirs(BASE_DIR)

model = tf.keras.models.load_model(model_dir)

title_preprocessor = tf.keras.layers.TextVectorization(max_tokens = VOCAB_SIZE,
                                                        output_sequence_length = MAX_LENGTH)

title_preprocessor.load_assets(vocab_dir)

## ------------------------------------------------------ ##
unk_counts = []

for row in cur.execute("SELECT title FROM news_articles"):
    sequence = title_preprocessor(row[0])

    unk_count = np.count_nonzero(sequence == 1)

    unk_counts.append(unk_count)

## ------------------------------------------------------ ##
ids = range(1, len(unk_counts) + 1)

plt.plot(ids, unk_counts)

## ------------------------------------------------------ ##
for row in cur.execute("SELECT id,title FROM news_articles WHERE ID > 100"):
    print(row)
