
# coding: utf-8

# In[1]:

#from __future__ import division
import os
import pandas as pd
import datetime as dt
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import scipy 
import pandas as pd


import os
import pandas as pd
import numpy as np
#import prep
import math
from statsmodels.stats.descriptivestats import sign_test

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn import metrics, svm, linear_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
#model builiding


from scipy import stats, interp
from scipy.stats import skew, kurtosis
from statsmodels.stats.descriptivestats import sign_test

#graphics
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt 
import seaborn as sns
# fix random seed for reproducibility

np.random.seed(42)


# In[ ]:

def calc_features(df, features):
    
    new_features=[]

    operation=['mean', 'max', 'std', 
               lambda x: (np.argmax(x.reset_index(drop=True))), #hier war vorher der Fehler, weil x nicht matrix!
               lambda x: (skew(x)), 
               lambda x: (kurtosis(x))]
    operation_name=['mean', 'max', 'std', 'argmax', 'skew', 'kurtosis']
 
    # for each conversation-part  
    #try:
    for part in set(df.conversation.dropna()):
        for unit in features: #for every feature
            print (unit)
               
            for i, op in enumerate(operation):
                feature_name=unit+'_'+operation_name[i]+'_'+str(part)
                df[feature_name]=df[df.conversation==part].groupby('vpn')[unit].transform(op)                
                new_features.append(feature_name)
                print (feature_name + "_has_been_calculated")
            
    return df, new_features

    # es entstehen NaNs dadurch, dass nur für jeden Abschnitt die jeweiligen Abschnitte berechnet werden.


# In[ ]:

def prepare_for_classify(df, features):

    #different feature spaces  
    df=df.groupby('vpn').mean().reset_index()
    
    X=np.array(df[features])

    # y=ASC-Diagnosis 
    y=np.array(df.asc)
    
    #vpn
    vpn=np.array(df.vpn)
    
    #shuffle
    index=np.array(np.random.choice(len(y), size=len(y), replace=False))
  
    X=X[index]
    vpn=vpn[index]
    y=y[index]
    
    
    return X, y, vpn
    


# In[ ]:

def classification(X, y, name, svm=False, n_hypercv=5):
    #Initialisierung verschiedener Variablen

    y_pred_rf = []
    y_pred_svm = []
    rf_parameter=[]
    svm_parameter=[]
    
    rf_pr=np.zeros([len(y),2])
    svm_pr=np.zeros([len(y),2])
    y_true=np.zeros(len(y)) 
    test_index=np.zeros(len(y)) 

    #Festlegen der mit Crossvalidierung zu tunenden Parameter

    tuned_rt_parameters = [{'max_depth':[1, 2, 4, 8, 16, 32, 64],
                           'min_samples_leaf':[1, 2, 4, 8, 16, 32, 64]}]
        
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    tuned_svm_parameters = [{'C':C_range, 'gamma':gamma_range}]  

    #Nested Cross-Validation
    i=0   
    loo = LeaveOneOut()
    
    #Scaling fuer SVM
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #X = scaler.fit_transform(X)

    for train, test in loo.split(X):
        print (train)
        
        #Random-Forest Classifier     
        rf_clf = GridSearchCV(RandomForestClassifier(n_estimators=1000, class_weight='balanced', n_jobs=-1), 
                                                     tuned_rt_parameters, cv=3,  n_jobs=-1, iid=False)
        rf_clf.fit(X[train], y[train]) #fitting                        
        y_pred_rf = np.append(y_pred_rf, rf_clf.predict(X[test])) #prediction
        probas_rf = rf_clf.predict_proba(X[test]) #class probability
        
        rf_pr[i]=np.array(probas_rf)
        rf_parameter=np.append(rf_parameter, rf_clf.best_params_)   
 
        #SVM
        if svm==True:     
            svm_clf = GridSearchCV(svm.SVC(class_weight=None, probability=True), 
                                   tuned_svm_parameters, cv=3, n_jobs=-1, iid=False) 

            svm_clf.fit(X_scaled[train], y[train]) #fitting
            y_pred_svm = np.append(y_pred_svm, svm_clf.predict(X_scaled[test])) #prediction
            probas_svm = svm_clf.predict_proba(X_scaled[test]) #class probability
            svm_pr[i]=np.array(probas_svm)
            svm_parameter=np.append(svm_parameter, svm_clf.best_params_)

        
        #Parameter der Classifier auf den Training-Test-Splits
        #print "Features sorted by their score:"
        #print sorted(zip(map(lambda x: round(x, 4), rf_clf.best_estimator_.feature_importances_), names), 
        #         reverse=True)           
            
                   
        y_true[i]=int(y[test])
        test_index[i]=int(test)
        i=i+1   

    #crossvalidated ROC-Curve
    roc_RF_name='RandomForest_'+name
    roc_graph(y_true, rf_pr[:, 1], roc_RF_name)
    
    #Evaluation der crossvalidierten Ergebnisse 
    print ('Crossvalidierte Ergebnisse fuer Random Forest')
    print (metrics.classification_report(y_true,y_pred_rf))
        
    results=pd.DataFrame([y_true, rf_pr, rf_parameter, test_index])
    
    if svm==True:
        roc_SVM_name='SVM_'+name
        roc_graph(y_true, svm_pr[:, 1], roc_SVM_name)
        results=pd.DataFrame([y_true, rf_pr, rf_parameter, svm_pr, svm_parameter, test_index])
        print ('Crossvalidierte Ergebnisse fuer Support Vector Machine')
        print (metrics.classification_report(y_true,y_pred_svm))  

    results.to_csv(name +'_Results.csv')
    
    return rf_pr, svm_pr, y_true, test_index, rf_parameter, svm_parameter


# In[ ]:

# Prediction based on AQ

def threshold(X, y, n_hypercv=5):
    #Initialisierung verschiedener Variablen

    y_pred_lr = []  
    lr_pr=np.zeros([len(y),2])
    y_true=np.zeros(len(y)) 
    test_index=np.zeros(len(y)) 


    tuned_lr_parameters = [{'C':[ 0.001, 0.01,  0.1, 1.,  5., 10., 20],  
                          }]
      
    i=0   
    loo = LeaveOneOut()

    for train, test in loo.split(X):
   
        y_true[i]=int(y[test])
        test_index[i]=int(test)

        #LR
        lr_clf = GridSearchCV(linear_model.LogisticRegression(class_weight='balanced'), 
                               tuned_lr_parameters) 
        lr_clf.fit(X[train], y[train])
        y_pred_lr = np.append(y_pred_lr, lr_clf.predict(X[test]))
        probas_lr = lr_clf.predict_proba(X[test])
        lr_pr[i]=np.array(probas_lr)
        
            
        i=i+1   

    roc_graph(y_true, lr_pr[:, 1], 'LogRec')
    
    #Evaluation der crossvalidierten Ergebnisse 
    print ('Crossvalidierte Ergebnisse fuer Logistic Regression')
    print (metrics.classification_report(y_true,y_pred_lr))
        
    results=pd.DataFrame([lr_pr, y_true, test_index])
    results.to_csv('AQ_LogRec_Results.csv')
    
    return lr_pr, y_true, test_index


# In[ ]:

#ROC-Curve for one classifier

def roc_graph(y_true, pr, name):
    fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_true, pr, drop_intermediate=False)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    plt.figure()
    plt.plot(fpr_nn, tpr_nn, lw=1, label='ROC (area = %0.2f)' % (roc_auc_nn))   
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')   
    plt.savefig('ROC_' +name+'.png')
    plt.show()   
    plt.close()


# In[ ]:

# ## Statistical Testing of the different prediction
def evaluate(y_true, predictions, asq, asq_index):
    i=0
    stat_key=[]
    stat_M=[]
    stat_Mp=[]
    stat_R=[]
    stat_Rp=[]
    
    for key in predictions:        
        pred=predictions[key]>0.5
        correcte_pred=(y_true==pred) # Prüft, ob korrekte Vorhersage durch Classifier
        correcte_mayority_pred=(np.ones(len(y_true))==y_true) #Baseline: Mehrheitsklasse zum vergleichen
        

        stat_key.append(key)
        M, p=sign_test((correcte_mayority_pred-correcte_pred), mu0=0) #returns M = (N(+) - N(-))/2 and p
        stat_results_M.append(M)
        stat_results_Mp.append(p)
        
        plt.figure()
        sns.regplot(predictions[key][asq_index], asq[asq_index])
        plt.savefig('Correlation'+ key +'.png')
        plt.show()
        plt.close()
        
        R, p = stats.pearsonr(predictions[key][asq_index], asq[asq_index])
        stat_results_R.append(R)
        stat_results_Rp.append(p)
        

    results=pd.DataFrame([stat_key, stat_M, stat_Mp, stat_R, stat_Rp])
    results.to_csv('RF_Stats.csv') 


# In[ ]:

#ROC-Curve for one classifier

def roc_graph(y_true, pr, name):
    fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_true, pr, drop_intermediate=False)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    plt.figure()
    plt.plot(fpr_nn, tpr_nn, lw=1, label='ROC (area = %0.2f)' % (roc_auc_nn))   
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')   
    plt.savefig('ROC_' +name+'.png')
    plt.show()   
    plt.close()
    
    #ROC-Curve to compare Classifier

def all_roc_graph(y_true, y_true_lr, lr_pr, predictions, name): 
    
    for key in predictions:
        fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_true, predictions[key], drop_intermediate=False)
        roc_auc_nn = auc(fpr_nn, tpr_nn)
        plt.figure()
        plt.plot(fpr_nn, tpr_nn, lw=1, label=key + ' :ROC (area = %0.2f)' % (roc_auc_nn))   
        
    fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_true_lr, lr_pr, drop_intermediate=False)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    plt.plot(fpr_nn, tpr_nn, lw=1, label='AQ-Baseline: ROC (area = %0.2f)' % (roc_auc_nn))   

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')   
    plt.savefig('ROC_RF_SVM' +'.png')
    plt.show()   
    plt.close()
    
#ROC-Curve to compare Features

def features_roc_graph(y_true, y_medium, y_true_big, y_true_mega, 
                       pr_svm_small, pr_rf_small,
                       pr_svm_medium, pr_rf_medium,  
                       pr_svm_big, pr_rf_big, 
                       pr_svm_mega, pr_rf_mega,
                       name_small, name_medium, name_big, name_mega, name):
    
    fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_true, pr_svm_small, drop_intermediate=False)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    plt.plot(fpr_nn, tpr_nn, lw=1, label='SVM' + name_small + ': ROC (area = %0.2f)' % (roc_auc_nn))   
    
    fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_true, pr_rf_small, drop_intermediate=False)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    plt.plot(fpr_nn, tpr_nn, lw=1, label='RF' + name_small +': ROC (area = %0.2f)' % (roc_auc_nn))   

    
    fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_medium, pr_svm_medium, drop_intermediate=False)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    plt.plot(fpr_nn, tpr_nn, lw=1, label='SVM' + name_medium + ': ROC (area = %0.2f)' % (roc_auc_nn))   
    
    fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_medium, pr_rf_medium, drop_intermediate=False)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    plt.plot(fpr_nn, tpr_nn, lw=1, label='RF' + name_medium +': ROC (area = %0.2f)' % (roc_auc_nn))   

    
    fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_true_big, pr_svm_big, drop_intermediate=False)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    plt.plot(fpr_nn, tpr_nn, lw=1, label='SVM ' + name_big + ': ROC (area = %0.2f)' % (roc_auc_nn))   
    
    fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_true_mega, pr_svm_mega, drop_intermediate=False)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    plt.plot(fpr_nn, tpr_nn, lw=1, label='RF'  + name_big + ':ROC (area = %0.2f)' % (roc_auc_nn))   
    
    fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_true_mega, pr_rf_mega, drop_intermediate=False)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    plt.plot(fpr_nn, tpr_nn, lw=1, label='RF'  + name_mega + ':ROC (area = %0.2f)' % (roc_auc_nn))   

    fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_true_mega, pr_svm_mega, drop_intermediate=False)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    plt.plot(fpr_nn, tpr_nn, lw=1, label='SVM'+  name_mega +': ROC (area = %0.2f)' % (roc_auc_nn))   


    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')   
    plt.savefig('ROC_features' + name + '.png')
    plt.show()    


# In[ ]:

# ## Statistical Testing of the different prediction
def evaluate(y_true, predictions, asq, asq_index):
    i=0
    stat_key=[]
    stat_M=[]
    stat_Mp=[]
    stat_R=[]
    stat_Rp=[]
    
    for key in predictions:        
        pred=predictions[key]>0.5
        correcte_pred=(y_true==pred) # Prüft, ob korrekte Vorhersage durch Classifier
        correcte_mayority_pred=(np.ones(len(y_true))==y_true) #Baseline: Mehrheitsklasse zum vergleichen
        

        stat_key.append(key)
        M, p=sign_test((correcte_mayority_pred-correcte_pred), mu0=0) #returns M = (N(+) - N(-))/2 and p
        stat_results_M.append(M)
        stat_results_Mp.append(p)
        
        plt.figure()
        sns.regplot(predictions[key][asq_index], asq[asq_index])
        plt.savefig('Correlation'+ key +'.png')
        plt.show()
        plt.close()
        
        R, p = stats.pearsonr(predictions[key][asq_index], asq[asq_index])
        stat_results_R.append(R)
        stat_results_Rp.append(p)
        

    results=pd.DataFrame([stat_key, stat_M, stat_Mp, stat_R, stat_Rp])
    results.to_csv('RF_Stats.csv') 


# In[ ]:

def rescaling(X, y, aq):
    scaler = MinMaxScaler(feature_range=(0, 1000))
    scaleraq = MinMaxScaler(feature_range=(0, 1))

    X=scaler.fit_transform(X[~np.isnan(aq), :])
    y=y[~np.isnan(aq)]
    aq=aq[~np.isnan(aq)]
    aq=scaler.fit_transform(aq[:, np.newaxis])
    
    return X, y, aq

def pca_on_X(X):
    pca = PCA(n_components=10)
    pca.fit(X)
    X_new=pca.transform(X) 
    return X_new

