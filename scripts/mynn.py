
# coding: utf-8

# In[ ]:

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import hyperband as hyperband 
#import prep
import seaborn as sns
from scipy import stats


from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics, svm, linear_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, StratifiedShuffleSplit, train_test_split

from scipy import interp
from scipy.stats import skew
from scipy.stats import kurtosis



# fix random seed for reproducibility
np.random.seed(42)

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dense, Input, LSTM, concatenate
from keras.layers.core import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping


# In[ ]:

# for IfI-Server
def ifi():
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

    import tensorflow as tf
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)


# In[ ]:

def prepare_CNN(df, features, version='normal'):
    if version=='audio':
        df=df[df.counter<16235].reset_index(drop=True)
        df['frame']=df.counter
        rows=16235
    else:
        rows=11310
        
    y=np.array(df.groupby('vpn').mean().asc)
    X = np.zeros((len(y),rows,len(features))).astype('float32')
    
    rescaledX = np.zeros((len(y),rows,len(features)))
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    for i, AU  in enumerate(features):
        feature_map=df.pivot(index='vpn', columns='frame', values=AU)
        feature_map=feature_map.fillna(0)  #Inplacement durch mean   
        X[:,:,i]=feature_map
        rescaledX[:,:,i]=scaler.fit_transform(feature_map) #nochmal überdenken

    #X=np.array(pd.DataFrame(X).dropna(axis=1, how='any'))
    asq=np.array(df.groupby('vpn').mean().asq)

    #shuffle
    index=np.array(np.random.choice(len(y), size=len(y), replace=False))
    X=X[index]
    rescaledX=rescaledX[index]
    y=y[index]
    asq=asq[index]
        
    asq_index=~np.isnan(asq)

    print ('shape of X' + str(X.shape))
    print ('class 0: ' + str(sum(y==0)))
    print ('class 1: ' + str(sum(y==1)))
    
    return X, rescaledX, y, asq, asq_index


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

from sklearn.grid_search import ParameterGrid


def cnn_simple(X,y, name):
    def create_model(nb_filters=3, nb_conv=2, pool=20, dropout=0.5, denseunits=2):
        model = Sequential()
        model.add(Convolution1D(nb_filters, nb_conv, activation='relu', 
                                input_shape=(X_train.shape[1], X_train.shape[2]), padding="same"))
        model.add(MaxPooling1D(pool))
        model.add(Dropout(dropout))
        model.add(Flatten()) 
        model.add(Dense(denseunits, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
        model.summary()
        return model

    history=[]
    acc_cv=[]
    acc_val_cv=[]
    loss_cv=[]
    loss_val_cv=[]
    y_true=[]
    probs_simple=[]
    i=0
    parameter=[]

    loo = LeaveOneOut()
    for train, test in loo.split(X):
        K.clear_session() #only for the lab for tensorflow

        X_train=X[train]
        X_test=X[test]
        y_train=y[train]
        y_test=y[test]

        model = KerasClassifier(build_fn=create_model, verbose=0)

        # define the grid search parameter
        nb_filters = [1, 2, 4, 6, 8]          
        nb_conv = [2, 4, 6, 8, 10]
        pool= [32, 64, 128, 256]    
        denseunits = [8, 16, 32, 64, 128, 256]
        dropout=[0.25, 0.5]
        param_grid = dict(nb_filters=nb_filters, nb_conv=nb_conv, pool=pool, dropout=dropout, denseunits=denseunits)
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
        
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid)    
        grid.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, 
                              callbacks=[earlyStopping])
        print ("Best: %f using %s" % (grid.best_score_, grid.best_params_))

        model=create_model(**grid.best_params_)
        
  
        history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, 
                              callbacks=[earlyStopping])
        
        score = model.evaluate(X_test, y_test)   
        probs_simple=np.append(probs_simple, model.predict_proba(X_test))
        y_true=np.append(y_true, y[test])

        print ("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        
        parameter.append(grid.best_params_)

    name=name+'simple_cnn'
    roc_graph(y_true, probs_simple, name)       
    
    results=pd.DataFrame([y_true, probs_simple, parameter])
    results.to_csv(name+'results.csv')
    
    return y_true, probs_simple


def cnn_stacked(X,y, name):
    def create_model(nb_filters=3, nb_conv=2, dropout=0.5, pool=20, denseSize=1):
        model = Sequential()
        model.add(Convolution1D(nb_filters, nb_conv, activation='relu', 
                                input_shape=(X_train.shape[1], X_train.shape[2]), padding="same"))
        model.add(MaxPooling1D(pool))
        model.add(Convolution1D(nb_filters, nb_conv, activation='relu', padding="same"))
        model.add(MaxPooling1D(pool))
        model.add(Dropout(dropout))
        model.add(Flatten()) 
        model.add(Dense(denseSize, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
        return model

    history=[]
    acc_cv=[]
    acc_val_cv=[]
    loss_cv=[]
    loss_val_cv=[]
    y_true=[]
    probs_simple=[]
    i=0
    parameter=[]
    
    loo = LeaveOneOut()
    for train, test in loo.split(X):
        K.clear_session() #only for the lab for tensorflow

        X_train=X[train]
        X_test=X[test]
        y_train=y[train]
        y_test=y[test]

        model = KerasClassifier(build_fn=create_model, verbose=0)

        # define the grid search parameter
        nb_filters = [1, 2, 4, 8]         
        nb_conv = [2, 4, 8, 16, 32]
        pool= [5, 25, 50]
        dropout= [0.25, 0.5]
        denseSize =  [8, 16, 32, 64, 128, 256]
        param_grid = dict(nb_conv=nb_conv, pool=pool, nb_filters=nb_filters, dropout=dropout, denseSize=denseSize)
       
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid)    
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
        
        grid.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, 
                              callbacks=[earlyStopping])
        
        print ("Best: %f using %s" % (grid.best_score_, grid.best_params_))

        model=create_model(**grid.best_params_)
          
        history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, 
                              callbacks=[earlyStopping])
        
        score = model.evaluate(X_test, y_test)   
        probs_simple=np.append(probs_simple, model.predict_proba(X_test))
        y_true=np.append(y_true, y[test])

        print ("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        #acc_cv=np.append(acc_cv, history.history['acc'])
        #acc_val_cv=np.append(acc_val_cv, history.history['val_acc'])

        #loss_cv.append(history.history['loss'])
        #loss_val_cv.append(history.history['val_loss'])

        parameter.append(grid.best_params_)

    name=name + 'cnn_stacked'
    roc_graph(y_true, probs_simple, name)       
    #visualise_acc(acc_cv, acc_val_cv, name)
    #visualise_loss(loss_cv, loss_val_cv, name)
    
    results=pd.DataFrame([y_true, probs_simple, parameter])
    results.to_csv(name+'_results.csv')
    
    return probs_simple


#Variation with seperated Pooling
#import hyperband as hyper
def cnn_pool(X, y, name):
    
    def getCNNModel(configuration):
        try:
            nb_filters = configuration['nb_filters']
            denseSize = configuration['denseSize']
            nb_conv = configuration['nb_conv']

            print (X_train_intro.shape)
            intro_input=Input(shape=(X_train_intro.shape[1], X_train_intro.shape[2]),  name='main_input0')
            intro_cnn=Convolution1D(nb_filters, nb_conv, activation='relu', padding="same")(intro_input)
            intro_pool=MaxPooling1D(X_train_intro.shape[1])(intro_cnn)  

            neutral_sp_input=Input(shape=(X_train_neutral_sp.shape[1], X_train_neutral_sp.shape[2]),  name='main_input1')
            neutral_sp_cnn=Convolution1D(nb_filters, nb_conv, activation='relu', padding="same")(neutral_sp_input)
            neutral_sp_pool=MaxPooling1D(X_train_neutral_sp.shape[1])(neutral_sp_cnn)  

            neutral_pr_input=Input(shape=(X_train_neutral_pr.shape[1], X_train_neutral_pr.shape[2]), name='main_input2')
            neutral_pr_cnn=Convolution1D(nb_filters, nb_conv, activation='relu', padding="same")(neutral_pr_input)
            neutral_pr_pool=MaxPooling1D(X_train_neutral_pr.shape[1])(neutral_pr_cnn) 

            joy_sp_input=Input(shape=(X_train_joy_sp.shape[1], X_train_joy_sp.shape[2]), name='main_input3')
            joy_sp_cnn=Convolution1D(nb_filters, nb_conv, activation='relu', padding="same")(joy_sp_input)
            joy_sp_pool=MaxPooling1D(X_train_joy_sp.shape[1])(joy_sp_cnn) 

            joy_pr_input=Input(shape=(X_train_joy_pr.shape[1], X_train_joy_pr.shape[2]),  name='main_input4')
            joy_pr_cnn=Convolution1D(nb_filters, nb_conv, activation='relu', padding="same")(joy_pr_input)
            joy_pr_pool=MaxPooling1D(X_train_joy_pr.shape[1])(joy_pr_cnn)  

            disgust_sp_input=Input(shape=(X_train_disgust_sp.shape[1], X_train_disgust_sp.shape[2]), name='main_input5')
            disgust_sp_cnn=Convolution1D(nb_filters, nb_conv, activation='relu', padding="same")(disgust_sp_input)
            disgust_sp_pool=MaxPooling1D(X_train_disgust_sp.shape[1])(disgust_sp_cnn) 

            disgust_pr_input=Input(shape=(X_train_disgust_pr.shape[1], X_train_disgust_pr.shape[2]), name='main_input6')
            disgust_pr_cnn=Convolution1D(nb_filters, nb_conv, activation='relu', padding="same")(disgust_pr_input)
            disgust_pr_pool=MaxPooling1D(X_train_disgust_pr.shape[1])(disgust_pr_cnn) 

            merged = concatenate([intro_pool, neutral_sp_pool, neutral_pr_pool, joy_sp_pool,
                                  joy_pr_pool, disgust_sp_pool, disgust_pr_pool],
                                 axis=1)

            model_flat = Flatten()(merged)             
            denselay = Dense(denseSize, activation='relu')(model_flat)
            predictions = Dense(1, activation='sigmoid')(denselay)

            model = Model(inputs=[intro_input, 
                                  neutral_sp_input, neutral_pr_input, 
                                  joy_sp_input, joy_pr_input, 
                                  disgust_sp_input, disgust_pr_input], outputs=predictions)
            model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])                 

            return model
        except:
            return None    

  
    def configGenerator(modelGenerator):
        while True:
            config = dict()    
            config['nb_filters'] = np.array([1, 2, 4, 6, 8, 10])[np.random.randint(6)] #erweitert um 10
            config['nb_conv'] = np.array([2, 4, 8, 16, 32])[np.random.randint(5)] 
            config['denseSize'] = np.array([4, 8, 16, 32, 64, 128, 256])[np.random.randint(7)] #erweitert um 256     

            model = modelGenerator(config)
            if model is None:
                print('error in model generation')
            if model is not None:
                return config

    def get_hyperparameter_configuration(configGenerator,modelGenerator):
        configurations = []
        nrconf=20 #changed from 10
        for i in np.arange(0,nrconf,1):
            new_config=configGenerator(modelGenerator)
            while new_config in configurations:
                new_config=configGenerator(modelGenerator)
            configurations.append(new_config)                
        return configurations
    
    from sklearn.model_selection import KFold
    
    def run_then_return_val_loss(config,modelGenerator,trainData,trainLabel):
        # parameter
        try:
            batch_size = 32 
            model = modelGenerator(config)
            if model != None:
                earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

                folds=3
                kf = KFold(len(trainLabel), n_folds=folds)
                i=0
                cv_score=np.zeros(folds)
                for train_index, test_index in kf:
                    model.fit(x=trainData[train_index], y=trainLabel[train_index],
                              nb_epoch=10,shuffle=True, initial_epoch=0,
                              batch_size=batch_size,
                              callbacks=[earlyStopping],
                              validation_data=(trainData[test_index], trainLabel[test_index]))
                    score = model.evaluate(trainData[test_index], trainLabel[test_index])
                    cv_score[i] = score[0]  
                    i=i+1
                    
            else:
                score = np.infty
            return score
        except:
            return np.infty
       

    def top_k(configurations,L,k):
        outConfigs = []
        sortIDs = np.argsort(np.array(L))
        for i in np.arange(0,k,1):
            outConfigs.append(configurations[int(sortIDs[int(i)])])
        return outConfigs 
    
    def hyperband(modelGenerator,
                  configGenerator,
                  trainData,trainLabel,
                  testData,testLabel,
                  outputFile=''):
        allLosses = []
        allConfigs = []
        L=[]
        configurations = get_hyperparameter_configuration(configGenerator,modelGenerator)
        
        for config in configurations:
            curLoss = run_then_return_val_loss(config,modelGenerator,
                                              trainData,trainLabel)
            L.append(curLoss)
            allLosses.append(curLoss)
            allConfigs.append(config)
            if outputFile != '':
                with open(outputFile, 'a') as myfile:
                    myfile.write(str(config) + '\t' + str(curLoss) +                                 '\t'  + '\n')

        bestConfig = top_k(allConfigs,allLosses,1)
        return (bestConfig[0],allConfigs,allLosses)

          
   
    probs_hyper_pool=np.zeros(len(y))
    y_true=np.zeros(len(y)) 
    parameter=[]
    
    i=0
    loo = LeaveOneOut()    
    

    for train, test in loo.split(X):
        K.clear_session() #only for the lab for tensorflow

        X_train=X[train]
                  
                  
                  
        X_test=X[test]
        y_train=y[train]
        y_test=y[test]

        X_train_intro=X_train[:, start[0]:end[0], :]

        X_train_neutral_sp=X_train[:, start[1]:end[1], :]
        X_train_neutral_pr=X_train[:, start[2]:end[2], :]

        X_train_joy_sp=X_train[:, start[3]:end[3], :]
        X_train_joy_pr=X_train[:, start[4]:end[4], :]

        X_train_disgust_sp=X_train[:, start[5]:end[5], :]
        X_train_disgust_pr=X_train[:, start[6]:end[6], :]

        X_test_intro=X_test[:, start[0]:end[0], :]

        X_test_neutral_sp=X_test[:, start[1]:end[1], :]
        X_test_neutral_pr=X_test[:, start[2]:end[2], :]

        X_test_joy_sp=X_test[:, start[3]:end[3], :]
        X_test_joy_pr=X_test[:, start[4]:end[4], :]

        X_test_disgust_sp=X_test[:, start[5]:end[5], :]
        X_test_disgust_pr=X_test[:, start[6]:end[6], :]

        X_train_pool=[X_train_intro, X_train_neutral_sp, X_train_neutral_pr,
                  X_train_joy_sp, X_train_joy_pr, X_train_disgust_sp, X_train_disgust_pr]

        X_test_pool=[X_test_intro,X_test_neutral_sp, X_test_neutral_pr,X_test_joy_sp, X_test_joy_pr, 
                                X_test_disgust_sp, X_test_disgust_pr]

        (bestConfig,allConfigs,allLosses) = hyperband(
                modelGenerator=getCNNModel,
                configGenerator=configGenerator,
                trainData=X_train_pool,
                trainLabel=y_train,
                testData=X_test_pool,
                testLabel=y_test,
                outputFile=name+'hyperbandOutput_seperated.txt')

        #print bestConfig
        #print np.min(allLosses)

        # calculate here now the pr for this trainingsexample 
        model=getCNNModel(bestConfig)
        
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
             
        history=model.fit(X_train_pool, y_train, validation_data=(X_test_pool, y_test),
                          epochs=10, batch_size=32, callbacks=[earlyStopping])

        #score = model.evaluate(X_test, y_test)   
        #probs_nn=np.append(probs_nn, model.predict_proba(X_test_pool))
        
        #try:
        probs_nn = model.predict(X_test_pool) #probs for ROC-Curve     
        #except:
        #    probs_nn=np.infty

        y_true[i]=int(y_test)
        probs_hyper_pool[i]=float(probs_nn)
        i=i+1
        print i
    
        parameter.append(bestConfig)

    roc_name=name+'_seperated_hyper'
    roc_graph(y_true, probs_hyper_pool, roc_name)     
    results=pd.DataFrame([y_true, probs_hyper_pool, parameter])
    results.to_csv(name+'results_hyper_sep.csv')
    
    return y_true, probs_hyper_pool

def lstm(X, y, name):    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #split the data
     
    parameter=[]
    history=[]
    acc_cv=[]
    acc_val_cv=[]
    loss_cv=[]
    loss_val_cv=[]
    y_true=[]
    probs_lstm=[]
    i=0
    loo = LeaveOneOut()
    for train, test in loo.split(X):
        
        K.clear_session() #only for the lab for tensorflow
        X_train=X[train]
        X_test=X[test]
        y_train=y[train]
        y_test=y[test]
        
        def create_model(outputunit=10, dropout=0.5, denseSize=2):
            model = Sequential()
            model.add(LSTM(outputunit, input_shape=X_train.shape[1:], dropout=dropout, recurrent_dropout=dropout))
            model.add(Dense(denseSize, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.summary()
            return model

        model = KerasClassifier(build_fn=create_model, verbose=0)

        # define the grid search parameter
        outputunit= [10, 25, 50, 75, 100]
        denseSize = [256]#{8, 16, 32, 64, 128, 256]    
        dropout= [0.25] #25, 0.5]
        
        outputunit= [10]
        
        param_grid = dict(outputunit=outputunit, dropout=dropout, denseSize=denseSize)
       
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=1)       
        
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
             
        grid.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, 
                              callbacks=[earlyStopping])

        model=create_model(**grid.best_params_)
                
     
        history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, 
                              callbacks=[earlyStopping])

        probs_lstm=np.append(probs_lstm, model.predict_proba(X_test))
        y_true=np.append(y_true, y[test])

        acc_cv=np.append(acc_cv, history.history['acc'])
        acc_val_cv=np.append(acc_val_cv, history.history['val_acc'])

        loss_cv.append(history.history['loss'])
        loss_val_cv.append(history.history['val_loss'])
        i=i+1
        
        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        parameter.append(grid.best_params_)
        
    name=name+'_lstm_'

    roc_graph(y_true, probs_lstm, name)       
    #visualise_acc(acc_cv, acc_val_cv, name)
    #visualise_loss(loss_cv, loss_val_cv, name)
    
    results=pd.DataFrame([y_true, probs_lstm, parameter])
    results.to_csv(name+'lstm.csv')
    
    return probs_lstm


# In[ ]:

def correlation_graph(pr, asq, test_index, y_true):
    cor_wert=stats.pearsonr(pr, asq[np.array(test_index, dtype=int)])
    sns.regplot(pr, asq[np.array(test_index, dtype=int)])
    plt.title('Classifier Confidence (NN_Pool) and AQ' + str(cor_wert))
    plt.xlabel('Confidence CNN')
    plt.ylabel('AQ')
    plt.legend(loc='lower right')   
    plt.savefig('Correlation.png')
    plt.show()
    plt.close()
    pred=pr>0.5
    print (stats.pearsonr(pred, y_true))
    
    
def all_roc_graph(y_true, predictions): #, probs_hyper, probs_hyper_pool):
    
    for key in predictions:
        fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_true, predictions[key], drop_intermediate=False)
        roc_auc_nn = auc(fpr_nn, tpr_nn)
        plt.figure()
        plt.plot(fpr_nn, tpr_nn, lw=1, label=key + ': ROC (area = %0.2f)' % (roc_auc_nn))   
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')   
    plt.savefig('ROC_NNs_ohne_hyper.png')
    plt.show()   
    plt.close()


# In[ ]:



