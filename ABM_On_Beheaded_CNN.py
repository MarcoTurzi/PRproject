#This file, written by the members of our MPR project group ourselves, contains code to apply...
#adversarial bias mitigation (ABM) on what we call a "beheaded CNN", which is a trained and fixed...
#CNN model which has been stripped of its top layer so that its final layer and output is a flattened
# array of node outputs.


from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.datasets import StructuredDataset, BinaryLabelDataset
import pandas as pd
from vgg_face import Beheaded_VGG_Face as VGG
import preproc as PP
import  tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def Run_ABM_on_VGG(dataRootDirectory,maxDataPerRace=100):
    """
    This method takes a beheaded trained CNN and learns its final layer using ABM learning.
    """
    dh = PP.DataHandler(dataRootDirectory)
    vgg = VGG.GetBeheadedVGG()
    DFtr= dh.get_transformed_feature_df(vgg,VGG.preprocess_img,type="train",maxDataPerRace= maxDataPerRace)

    DFtr['priv'] = DFtr['race'].map(lambda x: x == 'caucasian')
    DFtr= DFtr.drop(['race'],axis=1)
    SDtr = BinaryLabelDataset(favorable_label=1.0,unfavorable_label=0.0,df=DFtr,label_names=['female'],protected_attribute_names=['priv'])
    #SDte = BinaryLabelDataset(favorable_label=1.0,unfavorable_label=0.0,df=DFte,label_names=['female'],protected_attribute_names=['race'])

    priv_grp = [{'priv': 1}]
    unpriv_grp = [{'priv':0}]
    sess = tf.Session()

    # Without Bias mitigation
    plain_model = AdversarialDebiasing(privileged_groups = priv_grp,
                          unprivileged_groups = unpriv_grp,
                          scope_name='plain_classifier',
                          debias=False,
                          sess=sess)
    plain_model.fit(SDtr)

    # With bias mitigation
    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()
    debiased_mod = AdversarialDebiasing(privileged_groups = priv_grp,
                          unprivileged_groups = unpriv_grp,
                          scope_name='debiased_classifier',
                          debias=True,
                          sess=sess)
    debiased_mod.fit(SDtr)

    return plain_model, debiased_mod





