import tensorflow as tf
import matplotlib.pyplot as plotter_lib
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix



# Define the path to the train and test sets
train_path = 'final_train_dataset'
test_path_afro = 'new_data_test_set\\afro-american'
test_path_cauc = 'new_data_test_set\\caucasian'
test_path_asian = 'new_data_test_set\\asian'
test_path_indian = 'new_data_test_set\\indian'

# Create the train dataset
train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    image_size=(299,299),
    batch_size=32,
    seed=123,
    subset="both")




# Create the test dataset
test_ds_cauc = tf.keras.preprocessing.image_dataset_from_directory(
    test_path_cauc,
    image_size=(299,299),
    batch_size=32,
    seed=123)

test_ds_afro = tf.keras.preprocessing.image_dataset_from_directory(
    test_path_afro,
    image_size=(299,299),
    batch_size=32,
    seed=123)

test_ds_asian = tf.keras.preprocessing.image_dataset_from_directory(
    test_path_asian,
    image_size=(299,299),
    batch_size=32,
    seed=123)

test_ds_indian = tf.keras.preprocessing.image_dataset_from_directory(
    test_path_indian,
    image_size=(299,299),
    batch_size=32,
    seed=123)

model = tf.keras.Sequential()

# Create the base model
base_model = tf.keras.applications.ResNet50(weights='imagenet', 
                                         include_top=False,
                                         pooling="avg",
                                         input_shape=(299,299,3),
                                         classes=2)

# Freeze the base model
for each_layer in base_model.layers:

        each_layer.trainable=False

model.add(base_model)
# Add a new classifier on top
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(512, activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    min_delta=0.01, # minimium amount of change to count as an improvement
    patience=2, # how many epochs to wait before stopping
    restore_best_weights=True,
)

# Train the model
history = model.fit(train_ds,
    validation_data = val_ds,
    epochs=10,
    callbacks = [early_stopping])



labels = np.array([])
images = np.array([])
for i,l in test_ds_cauc.take(-1):
    labels = np.append(labels,l)
    images = np.append(images, i)

pred = model.predict(images)
pred = pred.reshape(-1,1)

print("CAUCASIAN")
print(classification_report(labels, pred))
print(confusion_matrix(labels, pred))

labels = np.array([])
images = np.array([])
for i,l in test_ds_afro.take(-1):
    labels = np.append(labels,l)
    images = np.append(images, i)

pred = model.predict(images)
pred = pred.reshape(-1,1)

print("AFRO-AMERICAN")
print(classification_report(labels, pred))
print(confusion_matrix(labels, pred))

labels = np.array([])
images = np.array([])
for i,l in test_ds_indian.take(-1):
    labels = np.append(labels,l)
    images = np.append(images, i)

pred = model.predict(images)
pred = pred.reshape(-1,1)

print("INDIAN")
print(classification_report(labels, pred))
print(confusion_matrix(labels, pred))

labels = np.array([])
images = np.array([])
for i,l in test_ds_asian.take(-1):
    labels = np.append(labels,l)
    images = np.append(images, i)

pred = model.predict(images)
pred = pred.reshape(-1,1)

print("ASIAN")
print(classification_report(labels, pred))
print(confusion_matrix(labels, pred))

history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['accuracy', 'val_accuracy']].plot()

plotter_lib.show()

history_df.loc[0:, ['loss', 'val_loss']].plot()
plotter_lib.show()

