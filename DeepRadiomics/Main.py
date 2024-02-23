#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn import metrics


# In[45]:


# An example: ALK mutation
modelPath_radiomics = './Models/ALK/3D_peritumoral/ALK_3D_peritumoral_random_forest.pkl'
modelPath_deep = './Models/ALK/deep/ALK_deep_random_forest.pkl'
modelPath_combination = './Models/ALK/deep_3D_peritumoral/ALK_deep_3D_peritumoral_random_forest.pkl'

testDataPath_radiomics = '../Data/Test/ALK/3D_peritumoral/radiomics_radscore_disease_test.xlsx'
testDataPath_deep = '../Data/Test/ALK/deep/radiomics_radscore_disease_test.xlsx'
testDataPath_combination = '../Data/Test/ALK/deep_3D_peritumoral/radiomics_radscore_disease_test.xlsx'


# In[46]:


# pklFilePath: a model object ending in 'pkl' format in the Models directory 
def loadModelFromPKL(pklFilePath):

    model = pickle.load(open(pklFilePath, 'rb'))

    return model


# In[53]:


def excutePredict(data, model, name):
    y_true = data['disease']
    X_test = data.drop('disease', axis=1)
    
    y_preds = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_probs[:, 1], pos_label=1)
    auc_score = roc_auc_score(y_true=y_true, y_score=y_probs[:, 1])
    accuray = metrics.accuracy_score(y_true=y_true, y_pred=y_preds)
    
    print('moedel_name:%s, auc_score:%s, accuray:%s' % (name, auc_score, accuray))


# In[62]:


def main():
    
    # radiomics model
    testData = pd.read_excel(testDataPath_radiomics, index_col=0)
    predictModel = loadModelFromPKL(modelPath_radiomics)
    excutePredict(testData, predictModel, 'radiomics model')
    
    # deep model
    testData = pd.read_excel(testDataPath_deep, index_col=0)
    predictModel = loadModelFromPKL(modelPath_deep)
    excutePredict(testData, predictModel, 'deep model')
    
    # combination model
    testData = pd.read_excel(testDataPath_combination, index_col=0)
    predictModel = loadModelFromPKL(modelPath_combination)
    excutePredict(testData, predictModel, 'combination model')
    


# In[63]:


if __name__ == '__main__':
    main()


# In[ ]:




