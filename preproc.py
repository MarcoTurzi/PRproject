#This code reads the face arg data and converts it to the format required for ABM
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img
import os

class DataHandler:
    def __init__(self, rootDirectory):
        self.rootD = rootDirectory
        self.trainDF = pd.read_csv(rootDirectory + '/train/train.csv')
        self.testDF = pd.read_csv(rootDirectory + '/test/test.csv')
    def get_transformed_feature_df(self, cnn, cnnPreproc, targetSize = (224,224),type="both", maxDataPerRace : int = 0):
        """

        :param cnn:
        :param cnnPreproc:
        :param type: can be either "train" or "test" to return train or test data only, or "both" to return both as a tuple.
        :return:
        """


        if not type in ["train","test","both"]:
            return None
        if type == "both":
            return self.get_transformed_feature_df(cnn,cnnPreproc, targetSize,"train",maxDataPerRace), \
                   self.get_transformed_feature_df(cnn,cnnPreproc, targetSize,"test",maxDataPerRace)
        print("Preprocessing for CNN model has started for " + type + " data...")
        df = self.trainDF.copy()
        if type == "test":
            df = self.testDF.copy()
        df.index = df['image_name']
        df.drop(['noGlasses','sunglasses','eyeglasses','age','hair_color','image_name'],axis=1,inplace=True)
        X = []
        indices = []

        for race in pd.unique(df['race']):
            nonExistent = 0
            i = 0
            for imgName in df[df['race']==race].index:

                path = self.rootD+'/'+type+'/'+df.loc[imgName,'race'] +'/'+imgName+'.jpg'
                if  not isinstance(path,str) or not os.path.exists(path):
                    nonExistent += 1
                    continue

                if maxDataPerRace > 0 and i > maxDataPerRace:
                    break
                img = load_img(path, target_size = targetSize)
                x = cnnPreproc(img)
                X.append(x)
                indices.append(imgName)

                i += 1
               # print(str(i))
            if nonExistent > 0:
                print("Warning: " + str(nonExistent) + " filenames were not found.")

        X = np.array(X)
        print("Preprocessing for CNN model has finished for " + type + " data...")
        print("CNN transformation has started for " + type + " data...")
        O = cnn.predict(X)
        del X
        dfRes = pd.DataFrame(O)
        del O
        dfRes.index = indices
        dfRes = dfRes.join(df)
        print("CNN transformation has finished for " + type + " data...")
        return dfRes







