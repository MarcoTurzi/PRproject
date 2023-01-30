# This file, partly written by the members of our MPR project group ourselves and partly taken from a demo
#by AIF360(https: // github.com / Trusted - AI / AIF360 / blob / master / examples / demo_adversarial_debiasing.ipynb), contains code to apply...
# adversarial bias mitigation (ABM) on what we call a "beheaded CNN", which is a trained and fixed...
# CNN model which has been stripped of its top layer so that its final layer and output is a flattened
# array of node outputs.
import os.path

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.datasets import StructuredDataset, BinaryLabelDataset
import pandas as pd
from vgg_face import Beheaded_VGG_Face as VGG
import preproc as PP
import tensorflow.compat.v1 as tf
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
import pickle

tf.disable_eager_execution()


def RunCompleteRoutineOnVGG(dataRootDirectory, maxDataPerRace=100, incorporate_train_data_in_evaluation=True, save=True,
                            saveFolderName="model"):
    """
    This method can be used to run the whole routine to preprocess data, use VGG-Face to extract features from it and use ABM to
    learn to map these features to gender labels. This method is written by ourselves.
    @dataRootDirectory: the directory of the extracted, unaltered FaceARG folder containing all train and test data.
     @maxDataPerRace: specifies the number of samples that is taken from the data for all 4 races. It is either an integer,
        in which case the same value is used for all races or it is a <string,int> dictionary specifying the number of samples
        per race distinctly.
    @incorporate_train_data: A boolean telling whether predictions on train data should be made and stored.
    @save: A boolean telling whether to save the ABM_Runner after running the routine.
    @saveFolderName: the name of the folder in which to save the pickled ABM_Runner if it should be saved
    @return: the ABM_Runner object containing all train data, test data and model predictions.
    """
    abm = ABM_Runner()
    abm.Run_ABM_on_VGG(dataRootDirectory, maxDataPerRace)
    abm.Evaluate_All()
    if save:
        abm.Save(dataRootDirectory + "/pickled_objects/" + saveFolderName)

    return abm


class ABM_Runner:
    def __init__(self):
        self.trained = False
        self.priv_grp = [{'priv': 1}]
        self.unpriv_grp = [{'priv': 0}]
        self.Pred_test_plain = None
        self.Pred_train_plain = None
        self.Pred_test_debiased = None
        self.Pred_train_debiased = None
        self.train_data = None
        self.test_data = None
        self.raceTrain = []
        self.raceTest = []

    def Run_ABM_on_VGG(self, dataRootDirectory, maxDataPerRace=100):
        """
        This method creates a VGG minus its final layers and learns its final layer using ABM learning and a predictor only.
        @dataRootDirectory: the directory of the extracted, unaltered FaceARG folder containing all train and test data.
        @maxDataPerRace: specifies the number of samples that is taken from the data for all 4 races. It is either an integer,
        in which case the same value is used for all races or it is a <string,int> dictionary specifying the number of samples
        per race distinctly.
        """
        dh = PP.DataHandler(dataRootDirectory)
        vgg = VGG.GetBeheadedVGG()
        DFtr, DFte = dh.get_transformed_feature_df(vgg, VGG.preprocess_img, type="both", maxDataPerRace=maxDataPerRace)

        DFtr['priv'] = DFtr['race'].map(lambda x: x == 'caucasian')
        self.raceTrain = DFtr['race']
        DFtr = DFtr.drop(['race'], axis=1)
        DFte['priv'] = DFte['race'].map(lambda x: x == 'caucasian')
        self.raceTest = DFte['race']
        DFte = DFte.drop(['race'], axis=1)

        # Converting the data to a BinaryLabelDataset, a subtype of a structured dataset. It is like a dataframe,
        # but additionally it stores which columns have special meaning, such as the labels and the protected
        # attributes. It is the format required by the AIF360 ABM class for training and testing the model.
        # For more on this class see: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.datasets.BinaryLabelDataset.html


        SDtr = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0, df=DFtr, label_names=['female'],
                                  protected_attribute_names=['priv'])
        SDte = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0, df=DFte, label_names=['female'],
                                  protected_attribute_names=['priv'])

        self.train_data = SDtr
        self.test_data = SDte

        # The following lines are taken and adapted from the Adversarial Debiasing demo by AIF360
        # See: https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_adversarial_debiasing.ipynb
        # For more on the adversarial debiasing model itself see:
        # https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/inprocessing/adversarial_debiasing.py

        sess = tf.Session()

        # Without Bias mitigation
        print("Training has started for the adversarial learning model without bias mitigation...")
        plain_model = AdversarialDebiasing(privileged_groups=self.priv_grp,
                                           unprivileged_groups=self.unpriv_grp,
                                           scope_name='plain_classifier',
                                           debias=False,
                                           sess=sess)
        plain_model.fit(SDtr)
        print("Training has finished for the adversarial learning model without bias mitigation...")
        self.plain_model = plain_model
        self.__store_predictions__(plain_model, False)

        # With bias mitigation
        print("Training has started for the adversarial learning model with bias mitigation...")
        sess.close()
        tf.reset_default_graph()
        sess = tf.Session()
        debiased_mod = AdversarialDebiasing(privileged_groups=self.priv_grp,
                                            unprivileged_groups=self.unpriv_grp,
                                            scope_name='debiased_classifier',
                                            debias=True,
                                            sess=sess)
        debiased_mod.fit(SDtr)
        self.debiased_model = debiased_mod
        print("Training has finished for the adversarial learning model with bias mitigation...")
        self.__store_predictions__(debiased_mod, True)

        self.plain_model = plain_model
        self.debiased_model = debiased_mod
        self.trained = True

        return plain_model, debiased_mod

    def __store_predictions__(self, model, debiased, incorporate_train_data=True):
        """
        This method stores the predictions made by the model on test and if specified also train data.
        This way the model itself does not need to be stored.
        @model: the model that is to make the predictions
        @debiased: a boolean telling whether the model passed is with (True) or without (False) ABM.
        @incorporate_train_data: A boolean telling whether predictions on train data should be made and stored.
        """
        message = "Started running fitted plain model for subsequent evaluation..."
        if debiased:
            message = message.replace("plain", "debiased")
        print(message)

        if debiased:
            self.Pred_test_debiased = model.predict(self.test_data)
            if incorporate_train_data:
                self.Pred_train_debiased = model.predict(self.train_data)
        else:
            self.Pred_test_plain = model.predict(self.test_data)
            if incorporate_train_data:
                self.Pred_train_plain = model.predict(self.train_data)
        message = "Finished running fitted plain model for subsequent evaluation..."
        if debiased:
            message = message.replace("plain", "debiased")
        print(message)

    def Evaluate(self, debiased):
        """
        This method provides and prints a collection of performance and fairness metrics for the model.
         @debiased: a boolean telling whether evaluation should be done on the results of the model with (True) or without (False) ABM.
        """
        if not self.trained:
            print("Cannot evaluate yet. Please train the ABM model first.")
            return
        privileged_groups = self.priv_grp
        unprivileged_groups = self.unpriv_grp
        Y_test = self.Pred_test_plain
        Y_train = self.Pred_train_plain
        if debiased:
            Y_test = self.Pred_test_debiased
            Y_train = self.Pred_train_debiased
        incorporate_train_data = False
        if (debiased and not self.Pred_train_debiased == None) or (not debiased and not self.Pred_train_plain == None):
            incorporate_train_data = True

        # The following lines are taken and adapted from the Adversarial Debiasing demo by AIF360
        # See: https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_adversarial_debiasing.ipynb

        caption = "#### Plain model - with debiasing - dataset metrics:"
        if not debiased:
            caption = caption.replace("with", "without")
        if incorporate_train_data:
            print(caption)
            metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(Y_train,
                                                                        unprivileged_groups=unprivileged_groups,
                                                                        privileged_groups=privileged_groups)

            print(
                "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())

        metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(Y_test,
                                                                   unprivileged_groups=unprivileged_groups,
                                                                   privileged_groups=privileged_groups)

        print(
            "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

        caption = "#### Plain model - with debiasing - classification metrics:"
        if not debiased:
            caption = caption.replace("with", "without")
        print(caption)
        classified_metric_nodebiasing_test = ClassificationMetric(self.test_data,
                                                                  Y_test,
                                                                  unprivileged_groups=unprivileged_groups,
                                                                  privileged_groups=privileged_groups)
        print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
        TPR = classified_metric_nodebiasing_test.true_positive_rate()
        TNR = classified_metric_nodebiasing_test.true_negative_rate()
        bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
        print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
        print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
        print(
            "Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
        print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
        print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())

    def Evaluate_All(self, includeIntersectionalResults=False):
        """
        This method calls other methods to perform evaluation on model with and without ABM and is implemented by ourselves.
        """
        if not self.trained:
            print("Cannot evaluate yet. Please train the ABM model first.")
            return
        if includeIntersectionalResults:
            self.Evaluate_Per_Race(False)
            self.Evaluate_Per_Race(True)
        self.Evaluate(False)
        self.Evaluate(True)

    def Evaluate_Per_Race(self, debiased):
        """
        This method provides  and prints accuracy, precission, recall and F1-score for all intersectional gender, race sub-populations.
        It is implemented by ourselves.
        @debiased: a boolean telling whether evaluation should be done on the model with (True) or without (False) ABM.
        """
        dataMod = self.Pred_test_plain
        if debiased:
            dataMod = self.Pred_test_debiased
        dataTrue = self.test_data

        for race in pd.unique(self.raceTest):
            df = pd.DataFrame(index=['female', 'male'])
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            Y = dataTrue.labels[:, 0]
            Y_hat = dataMod.labels[:, 0]
            Z = self.raceTest.array
            for i in range(len(Y)):
                if Z[i] != race:
                    continue
                if Y_hat[i] == 1.0:
                    if Y[i] == 1.0:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if Y[i] == 0.0:
                        TN += 1
                    else:
                        FN += 1
            TP_M = TN
            TN_M = TP
            FP_M = FN
            FN_M = FP
            df['accuracy'] = [(TP + TN) / (TP + TN + FP + FN), (TP_M + TN_M) / (TP_M + TN_M + FP_M + FN_M)]
            df['recall'] = [TP / (TP + FN), TP_M / (TP_M + FN_M)]
            df['precision'] = [TP / (TP + FP), TP_M / (TP_M + FP_M)]
            df['f1'] = [2 * TP / (2 * TP + FP + FN), 2 * TP_M / (2 * TP_M + FP_M + FN_M)]
            caption = "Intersectional results for biased model on " + race + " data:"
            if debiased:
                caption = caption.replace("biased", "unbiased")
            print(caption)
            print(df)

    def Save(self, directory):
        """
        This method saves all important data stored in the ABM runner so it can be retrieved for further analysis.
        @directory: specifies the path at which to save this.
        Method implemented by ourselves
        """
        if not os.path.exists(directory):
            os.mkdir(directory)
        df, dict = self.Pred_train_plain.convert_to_dataframe()
        df.to_pickle(directory + '/train_plain.pkl')
        file = open(directory + '/label_dict.pkl', 'wb')
        pickle.dump(dict, file)
        file.close()
        df, _ = self.Pred_test_plain.convert_to_dataframe()
        df.to_pickle(directory + '/test_plain.pkl')
        df, _ = self.Pred_train_debiased.convert_to_dataframe()
        df.to_pickle(directory + '/train_debiased.pkl')
        df, _ = self.Pred_test_debiased.convert_to_dataframe()
        df.to_pickle(directory + '/test_debiased.pkl')
        df, _ = self.test_data.convert_to_dataframe()
        df.to_pickle(directory + '/test_data.pkl')
        df, _ = self.train_data.convert_to_dataframe()
        df.to_pickle(directory + '/train_data.pkl')
        self.raceTest.to_pickle(directory + '/test_race.pkl')
        self.raceTrain.to_pickle(directory + '/train_race.pkl')

    def Create_from_folder(directory):
        """
        This method loads and returns a ABM_Runner object stored with the ABM_Runner.Save method.
        @directory: the path to the folder from which to load the ABM_Runner
        returns: the loaded ABM_Runner object
        This method is implemented by ourselves.
        """
        abm = ABM_Runner()
        file = open(directory + "/label_dict.pkl", "rb")
        labelDict = pickle.load(file)
        file.close()
        df = pd.read_pickle(directory + '/train_plain.pkl')

        abm.Pred_train_plain = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0,
                                                  df=pd.read_pickle(directory + '/train_plain.pkl'),
                                                  label_names=labelDict['label_names'],
                                                  protected_attribute_names=labelDict['protected_attribute_names'])
        abm.Pred_test_plain = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0,
                                                 df=pd.read_pickle(directory + '/test_plain.pkl'),
                                                 label_names=labelDict['label_names'],
                                                 protected_attribute_names=labelDict['protected_attribute_names'])
        abm.Pred_train_debiased = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0,
                                                     df=pd.read_pickle(directory + '/train_debiased.pkl'),
                                                     label_names=labelDict['label_names'],
                                                     protected_attribute_names=labelDict['protected_attribute_names'])
        abm.Pred_test_debiased = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0,
                                                    df=pd.read_pickle(directory + '/test_debiased.pkl'),
                                                    label_names=labelDict['label_names'],
                                                    protected_attribute_names=labelDict['protected_attribute_names'])
        abm.test_data = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0,
                                           df=pd.read_pickle(directory + '/test_data.pkl'),
                                           label_names=labelDict['label_names'],
                                           protected_attribute_names=labelDict['protected_attribute_names'])
        abm.train_data = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0,
                                            df=pd.read_pickle(directory + '/train_data.pkl'),
                                            label_names=labelDict['label_names'],
                                            protected_attribute_names=labelDict['protected_attribute_names'])
        abm.raceTest = pd.read_pickle(directory + '/test_race.pkl')
        abm.raceTrain = pd.read_pickle(directory + '/train_race.pkl')
        abm.trained = True
        return abm






