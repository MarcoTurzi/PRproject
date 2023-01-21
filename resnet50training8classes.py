import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Define the path to the train and test sets
train_path = 'F:\\FaceRecognition\\PRproject\\utils\\new_data_set\\FaceARG\\train'
test_path = 'F:\\FaceRecognition\\PRproject\\utils\\new_data_test_set\\FaceARG\\test'

# Create the train dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    batch_size=128)
train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=8)))

# Create the test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    batch_size=128)
test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y, depth=8)))

# Create the base model
base_model = keras.applications.ResNet50(weights='imagenet', 
                                         include_top=False)

# Freeze the base model
base_model.trainable = False

# Add a new classifier on top
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x) 
x = keras.layers.Dense(500, activation='relu')(x)
x = keras.layers.Dense(500, activation='relu')(x)
predictions = keras.layers.Dense(8, activation='softmax')(x)

# Create the final model
model = keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
schedule = tf.keras.optimizers.schedules.CosineDecay(1e-2, 100, alpha=1e-2, name=None)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=schedule),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_ds, epochs=100, verbose = 1)

model.save('F:\\FaceRecognition\\PRproject\\utils\\model\\model100')


# Make predictions on the test set

acc_test,len = 0., 0.
for i,(x,y) in enumerate(test_ds):
  y_pred = tf.argmax(model.predict(x),axis=1)
  acc_test += (y_pred==y).sum()
  len += y.shape[0]

print('Accuracy (test set) :  {} %'.format(acc_test/len))