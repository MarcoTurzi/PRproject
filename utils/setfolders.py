import pandas as pd
import os
import shutil

df = pd.read_csv("FaceARG\\test\\test.csv")

male = df[df["female"] == 0]
female = df[df["female"] == 1]



# Define the prefixes for the male and female image arrays
male_prefixes = male["image_name"].to_numpy()
female_prefixes = female["image_name"].to_numpy()
print(male_prefixes)
print(female_prefixes)

# Define the paths for the 4 folders containing face images
folders = ["FaceARG\\test\\afro-american", "FaceARG\\test\\asian", "FaceARG\\test\\caucasian", "FaceARG\\test\\indian"]

for folder in folders:
    # Define the paths for the male and female directories
    male_dir = f"new_data_set\\{folder}\\male"
    female_dir = f"new_data_set\\{folder}\\female"

    # Create the male and female directories
    os.makedirs(male_dir, exist_ok=True)
    os.makedirs(female_dir, exist_ok=True)

    for file_name in os.listdir(folder):
        file_name = file_name.replace(".jpg","")
        if file_name in male_prefixes:
            shutil.copy2(f"{folder}/{file_name}.jpg", f"{male_dir}/{file_name}.jpg")
        elif file_name in female_prefixes:
            shutil.copy2(f"{folder}/{file_name}.jpg", f"{female_dir}/{file_name}.jpg")
