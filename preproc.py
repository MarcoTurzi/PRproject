#This code reads the face arg data and converts it to the format required for ABM
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img
import os
import pickle
import gc

class DataHandler:
    def __init__(self, rootDirectory):
        self.rootD = rootDirectory
        self.trainDF = pd.read_csv(rootDirectory + '/train/train.csv')
        self.testDF = pd.read_csv(rootDirectory + '/test/test.csv')
        if not os.path.exists(rootDirectory + '/pickled_objects'):
            os.mkdir(rootDirectory + '/pickled_objects')
    def get_transformed_feature_df(self, cnn, cnnPreproc, targetSize = (224,224),type="both", maxDataPerRace = 0):
        """

        @param cnn: The convolutional neural network model that will be used to extract a high level representation vector
        from the images. It should have a predict function that takes a (preprocessed) multidimensional array and outputs a vector/
        @param cnnPreproc: The function used to preprocess images before they can be processed by the cnn.
        @param targetSize: The dimensions the input array of the CNN should have.
        @param type: This is a string specifying the type of data that will be extracted. Its value can either be "train" for
        extracting only training data, test for extracting only testing data, or "both" for extracting both as a tuple.
        @param maxDataPerRace: this parameter determines the size of the subset of data that is used by specifying the
        number of images that is included per race (per condition). If this value is an int, this same value will be used
        for all races. If it is a <string, int> dictionary, with a key for each race, each race will have its own maximum
        image value. If any value is zero or lower, all data will be used.
        @return: A dataframe consisting of several data samples each of which consists of information on race, gender and the
        elements of a high level vector representing the image. If type is "both", a tuple of a train set data frame and a
        test set data frame will be returned.
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
            print('Preprocessing data for race label "'+ race +'"')
            nonExistent = 0
            i = 0
            max = maxDataPerRace
            if isinstance(maxDataPerRace,dict):
                max = maxDataPerRace[race]
            for imgName in df[df['race']==race].index:
                path = self.rootD+'/'+type+'/'+df.loc[imgName,'race'] +'/'+imgName+'.jpg'
                if  not isinstance(path,str) or not os.path.exists(path):
                    nonExistent += 1
                    continue

                if maxDataPerRace > 0 and i > max:
                    break
                img = load_img(path, target_size = targetSize)
                x = cnnPreproc(img)
                del img
                X.append(x)
                indices.append(imgName)

                i += 1
                #print(str(i))
                #gc.collect()
            if nonExistent > 0:
                print("Warning: " + str(nonExistent) + " filenames were not found.")

        X = np.array(X)
        print("Preprocessing for CNN model has finished for " + type + " data...")
        print("CNN transformation has started for " + type + " data...")





        O = cnn.predict(X)
        del X




        dfRes = pd.DataFrame(O)

        dfRes.index = indices
        dfRes = dfRes.join(df)

        del O
        dir = self.rootD + "/pickled_objects/" + "ABM_"+type+"_data.pkl"
        if os.path.exists(dir):
            os.remove(dir)
        dfRes.to_pickle(dir)

        print("CNN transformation has finished for " + type + " data...")
        return dfRes







