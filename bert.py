import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optmizer

AUTOTUNE = tf.data.AUTOTUNE
BATCH = 64
seed = 42
NUM_EPOCHS = 3
init_lr = 3e-5

#Creating subsets of data
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train', batch_size=BATCH, validation_split=0.2,
    subset='training', seed=seed)
class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train', batch_size=BATCH, validation_split=0.2,
    subset='validation', seed=seed)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/test', batch_size=BATCH)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#BERT Initialization
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * NUM_EPOCHS
num_warmup_steps = int(0.1*num_train_steps)

model = build_classifier_model()
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()
optimizer = optimization.create_optimizer(init_lr=init_lr, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, optimizer_type='adamw')
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print(f'Training model with {tfhub_handle_encoder}')

model.fit(train_ds, epochs=NUM_EPOCHS, validation_data = val_ds)
model.evaluate(test_ds)