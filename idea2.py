from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import image_dataset_from_directory

# Load the datasets
gender_data = image_dataset_from_directory("path/to/gender_data", labels='inferred', image_size=(224, 224))
race_data = image_dataset_from_directory("path/to/race_data", labels='inferred', image_size=(224, 224))

# Create the ResNet50 models for gender and race classification
gender_model = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3), classes=2)
race_model = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3), classes=5)

# Define the gradient reversal layer function
def gradient_reversal_layer(scalar):
    @tf.function
    def func(x):
        y = x * scalar
        return y
    return tf.keras.layers.Lambda(func)

# Create the new model that combines both gender and race models
inputs = layers.Input(shape=(224, 224, 3))
gender_output = gender_model(inputs)
race_output = gradient_reversal_layer(scalar=-1)(race_model(inputs))

# Use the gender output as the final output of the new model
model = models.Model(inputs=inputs, outputs=gender_output)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(gender_data, gender_data.labels, epochs=10)
