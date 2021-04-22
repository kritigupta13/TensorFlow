import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optmizer
import os
import sys
import json

print("Arguments count: ", len(sys.argv))
fname = sys.argv[1]
index = sys.argv[2]
print(fname, index)

AUTOTUNE = tf.data.AUTOTUNE
BATCH = 32
seed = 42
NUM_EPOCHS = 1
init_lr = 3e-5

with open(fname) as f:
  data = json.load(f)
data['task']['index'] = index
os.environ['TF_CONFIG'] = json.dumps(data)

num_workers = len(data['cluster']['worker'])
global_batch_size = BATCH * num_workers

#BERT Initialization
tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    #Creating subsets of data
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train', batch_size=BATCH, validation_split=0.2,subset='training', seed=seed)
    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train', batch_size=BATCH, validation_split=0.2,subset='validation', seed=seed)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/test', batch_size=BATCH)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * NUM_EPOCHS
    num_warmup_steps = int(0.1*num_train_steps)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    model = tf.keras.Model(text_input, net)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    optimizer = optimization.create_optimizer(init_lr=init_lr, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, optimizer_type='adamw')
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print('Training model with ', tfhub_handle_encoder)

model.fit(train_ds, epochs=NUM_EPOCHS, validation_data = val_ds)
model.evaluate(test_ds)