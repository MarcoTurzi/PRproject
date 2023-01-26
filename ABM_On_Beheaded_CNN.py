#This file, written by the members of our MPR project group ourselves, contains code to apply...
#adversarial bias mitigation (ABM) on what we call a "beheaded CNN", which is a trained and fixed...
#CNN model which has been stripped of its top layer so that its final layer and output is a flattened
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
import  pickle

tf.disable_eager_execution()

def RunCompleteRoutineOnVGG(dataRootDirectory, maxDataPerRace=100,incorporate_train_data_in_evaluation=True,  save = True, saveFolderName = "model"):
    """

    @param dataRootDirectory:
    @param maxDataPerRace:
    @param incorporate_train_data_in_evaluation:
    @param pickleABM_Runner:
    @return:
    """
    abm = ABM_Runner()
    abm.Run_ABM_on_VGG(dataRootDirectory,maxDataPerRace)
    abm.Evaluate_All()
    if save:
        abm.Save(dataRootDirectory+"/pickled_objects/"+saveFolderName)
    # WIP
    #if pickleABM_Runner:
     #   file = open(dataRootDirectory+'/pickled_objects/ABM_Runner_'+str(maxDataPerRace)+'_per_race.pkl','wb')
      #  pickle.dump(abm,file)
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
        This method takes a beheaded trained CNN and learns its final layer using ABM learning.
        """
        dh = PP.DataHandler(dataRootDirectory)
        vgg = VGG.GetBeheadedVGG()
        DFtr, DFte  = dh.get_transformed_feature_df(vgg,VGG.preprocess_img,type="both",maxDataPerRace= maxDataPerRace)

        DFtr['priv'] = DFtr['race'].map(lambda x: x == 'caucasian')
        self.raceTrain = DFtr['race']
        DFtr= DFtr.drop(['race'],axis=1)
        DFte['priv'] = DFte['race'].map(lambda x: x == 'caucasian')
        self.raceTest = DFte['race']
        DFte = DFte.drop(['race'], axis=1)

        SDtr = BinaryLabelDataset(favorable_label=1.0,unfavorable_label=0.0,df=DFtr,label_names=['female'],protected_attribute_names=['priv'])
        SDte = BinaryLabelDataset(favorable_label=1.0,unfavorable_label=0.0,df=DFte,label_names=['female'],protected_attribute_names=['priv'])

        self.train_data = SDtr
        self.test_data = SDte


        sess = tf.Session()

        # Without Bias mitigation
        print("Training has started for the adversarial learning model without bias mitigation...")
        plain_model = AdversarialDebiasing(privileged_groups = self.priv_grp,
                              unprivileged_groups = self.unpriv_grp,
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
        debiased_mod = AdversarialDebiasing(privileged_groups = self.priv_grp,
                              unprivileged_groups = self.unpriv_grp,
                              scope_name='debiased_classifier',
                              debias=True,
                              sess=sess)
        debiased_mod.fit(SDtr)
        self.debiased_model = debiased_mod
        print("Training has finished for the adversarial learning model with bias mitigation...")
        self.__store_predictions__(debiased_mod,True)

        self.plain_model = plain_model
        self.debiased_model = debiased_mod
        self.trained = True

        return plain_model, debiased_mod

    def __store_predictions__(self, model, debiased, incorporate_train_data=True):
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

    def Evaluate_All(self):
        if not self.trained:
            print("Cannot evaluate yet. Please train the ABM model first.")
            return
        self.Evaluate(False)
        self.Evaluate(True)

    def Save(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
        df, dict = self.Pred_train_plain.convert_to_dataframe()
        df.to_pickle(directory+'/train_plain.pkl')
        file = open(directory + '/label_dict.pkl','wb')
        pickle.dump(dict, file)
        file.close()
        df,_ = self.Pred_test_plain.convert_to_dataframe()
        df.to_pickle(directory+'/test_plain.pkl')
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
        abm = ABM_Runner()
        file = open(directory + "/label_dict.pkl", "rb")
        labelDict = pickle.load(file)
        file.close()
        df = pd.read_pickle(directory + '/train_plain.pkl')

        abm.Pred_train_plain = BinaryLabelDataset(favorable_label=1.0,unfavorable_label=0.0,
                                                  df=pd.read_pickle(directory + '/train_plain.pkl'),label_names=labelDict['label_names'],
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






