from tensorflow import keras
from keras.utils import image_dataset_from_directory
import pandas as pd
import os

df = pd.read_csv("FaceARG\\test\\test.csv")

male = df[df["female"] == 0]
female = df[df["female"] == 1]

print(male["image_name"].to_numpy())
print(female["image_name"].to_numpy())

# Define the prefixes for the male and female image arrays
male_prefixes = male["image_name"].to_numpy()
female_prefixes = female["image_name"].to_numpy()

# Define the paths for the 4 folders containing face images
folders = ["FaceARG\\train\\afro-american", "FaceARG\\train\\asian", "FaceARG\\train\\caucasian", "FaceARG\\train\\indian"]

for folder in folders:
    # Define the paths for the male and female directories
    male_dir = f"{folder}\\male"
    female_dir = f"{folder}\\female"

    # Create the male and female directories
    os.makedirs(male_dir, exist_ok=True)
    os.makedirs(female_dir, exist_ok=True)

    for file_name in os.listdir(folder):
        if any(prefix in file_name for prefix in male_prefixes):
            os.rename(f"{folder}/{file_name}", f"{male_dir}/{file_name}")
        elif any(prefix in file_name for prefix in female_prefixes):
            os.rename(f"{folder}/{file_name}", f"{female_dir}/{file_name}")
