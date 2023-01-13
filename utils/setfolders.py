import pandas as pd
import os

df = pd.read_csv("FaceARG\\train\\train.csv")

male = df[df["female"] == 0]
female = df[df["female"] == 1]



# Define the prefixes for the male and female image arrays
male_prefixes = male["image_name"].to_numpy()
female_prefixes = female["image_name"].to_numpy()
print(male_prefixes)
print(female_prefixes)

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
        if any(prefix in file_name for prefix in male_prefixes if not type(prefix) == type(9.8)):
            os.rename(f"{folder}/{file_name}", f"{male_dir}/{file_name}")
        elif any(prefix in file_name for prefix in female_prefixes if not type(prefix) == type(9.8) ):
            os.rename(f"{folder}/{file_name}", f"{female_dir}/{file_name}")
