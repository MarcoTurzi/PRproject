import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Define the path to the train and test sets
train_path = 'new_data_set\\afro-american'
test_path = 'new_data_test_set\\afro-american'

# Create the train dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    batch_size=32)

# Create the test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    batch_size=32)

# Create the base model
base_model = keras.applications.ResNet50(weights='imagenet', 
                                         include_top=False)

# Freeze the base model
base_model.trainable = False

# Add a new classifier on top
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x) 
predictions = keras.layers.Dense(1, activation='sigmoid')(x)

# Create the final model
model = keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_ds, epochs=10)

# Make predictions on the test set
y_pred = model.predict(test_ds)
y_pred = y_pred > 0.5

# Compute the classification report and confusion matrix
print(classification_report(test_ds.labels, y_pred))
print(confusion_matrix(test_ds.labels, y_pred))
