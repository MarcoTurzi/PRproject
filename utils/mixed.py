import os
import shutil
import pandas as pd 

df = pd.read_csv("FaceARG\\train\\train.csv")

male_df = df[df["female"] == 0]
female_df = df[df["female"] == 1]

male_prefixes = male_df["image_name"].to_numpy()
female_prefixes = female_df["image_name"].to_numpy()

source_folders = ["FaceARG\\train\\afro-american","FaceARG\\train\\asian","FaceARG\\train\\caucasian","FaceARG\\train\\indian"]
destination_folder = "mixed_train_dataset"

#create destination folders
os.makedirs(destination_folder + "/male")
os.makedirs(destination_folder + "/female")

# loop through the source folders
for folder in source_folders:
    files = os.listdir(folder)
    male_count = 0
    female_count = 0
    # loop through the files in the current folder
    for file in files:
        file = file.replace(".jpg","")
        # check if the file name starts with a male prefix
        if file in male_prefixes:
            # check if the male count is less than 2000
            if male_count < 2000:
                # move the file to the destination folder
                shutil.copy(folder + "/" + file +".jpg", destination_folder + "/male")
                male_count += 1
        # check if the file name starts with a female prefix
        elif file in female_prefixes:
            # check if the female count is less than 2000
            if female_count < 2000:
                # move the file to the destination folder
                shutil.copy(folder + "/" + file +".jpg", destination_folder + "/female")
                female_count += 1
