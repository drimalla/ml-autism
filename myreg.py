
# coding: utf-8

# In[ ]:

def prepare_for_regress_RF(df, df_audio, study, goal, features):

       
    #different feature spaces  
    
    df=pd.merge(df.groupby('vpn').mean(), df_audio.groupby('vpn').mean(), how='inner', on='vpn')  
    
    X = np.array(df.groupby('vpn').mean()[features_AU_intense + features_AU_occurence + features_gaze + features_audio])

    if study=='charite':

        if goal=='aq':   
            df=df[~df.asq.isnull()].reset_index(drop=True)
            y=np.array(df.groupby('vpn').mean().asq)
            
        if goal=='NT':   
            df=df[~df.asq.isnull()].reset_index(drop=True)
            df=df[df.ados_total==-97].reset_index(drop=True)
            y=np.array(df.groupby('vpn').mean().asq)

        if goal=='ADOS':        
            df=df[~df.ados_total.isnull()].reset_index(drop=True)
            df=df[df.ados_total!=-97].reset_index(drop=True)
            y=np.array(df.groupby('vpn').mean().ados_total) 
                    
        if goal=='ADOS_social': 
            df=df[~df.ados_total.isnull()].reset_index(drop=True)
            df=df[df.ados_total!=-97].reset_index(drop=True)
            y=np.array(df.groupby('vpn').mean().ados_social)
    
        if goal=='ADOS_communication': 
            df=df[~df.ados_total.isnull()].reset_index(drop=True)
            df=df[df.ados_total!=-97].reset_index(drop=True)
            y=np.array(df.groupby('vpn').mean().ados_commu)
            
                    
        if goal=='ADI': 
            df=df[~df.adir_social.isnull()].reset_index(drop=True)
            df=df[df.adir_social>0].reset_index(drop=True)
            y=np.array(df.groupby('vpn').mean().adir_social)


        diagnosis=np.array(df.groupby('vpn').mean()['asq'])
        
    if study=='mb':
        df=df[~df.AQsum.isnull()].reset_index(drop=True)
        df=df[df.AQsum>0].reset_index(drop=True)
        #df=df[df.AQsum<17].reset_index(drop=True)
        y=np.array(df.groupby('vpn').mean().AQsum)
        diagnosis=np.array((df.groupby('vpn').mean().AQsum)>17)
   
    def shuffle(X,y, diagnosis):
        
        index=np.array(np.random.choice(len(y), size=len(y), replace=False))
        
        X=X[index]
        y=y[index]
        diagnosis=diagnosis[index]

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_rescaled = scaler.fit_transform(X)
    
        return X, X_rescaled, y, diagnosis

    X, X_rescaled, y, diagnosis = shuffle(X, y, diagnosis)   
    
    return X, X_rescaled, y, diagnosis


# In[ ]:

def regression(X, y, name='name'):
    #Initialisierung verschiedener Variablen
    rf_parameter=[]
    svm_parameter=[]
    
    y_pred_svr = []
    
    y_pred_tree = []
    
    y_true=np.zeros(len(y)) 
    
    y_base=np.zeros(len(y)) 
       
    #plt.figure(1)

    #Nested Cross-Validation
    i=0   
    loo = LeaveOneOut()

    for train, test in loo.split(X):
        
        y_true[i]=int(y[test])
        y_base[i]=np.mean(y[train])
            
        tree = GridSearchCV(RandomForestRegressor(n_estimators=1000), cv=3,
                   param_grid={"max_depth": [2, 4, 5, 10, 15, 20]})
        tree.fit(X[train], y[train])
        
        y_pred_tree = np.append(y_pred_tree, tree.predict(X[test])) 
        
        svm_clf =  GridSearchCV(SVR(epsilon=0.2,  kernel='rbf'), cv=3,
                   param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10., 100],
                                'gamma': [0.001, 0.01, 0.1, 1.0, 10., 100]})
        svm_clf.fit(X[train], y[train])
        y_pred_svr = np.append(y_pred_svr, svm_clf.predict(X[test])) 
   
        rf_parameter=np.append(rf_parameter, tree.best_params_)   
        svm_parameter=np.append(svm_parameter, svm_clf.best_params_)
        i=i+1   
     
    #if y_pred_tree=

    print ('Crossvalidierte Ergebnisse fuer RFR')
    print (math.sqrt(mean_squared_error(y_true, y_pred_tree)))
    print ('Mean_Absolute_Error:')
    errors=np.abs(y_true-y_pred_tree)
    print (np.mean(errors))
    print ('Standard Deviation of the Error:')
    print (np.std(errors))
             
    print ('Crossvalidierte Ergebnisse fuer SVR')
    print (math.sqrt(mean_squared_error(y_true, y_pred_svr)))    
    print ('Mean_Absolute_Error:')
    errors=np.abs(y_true-y_pred_svr)
    print (np.mean(errors))
    print ('Standard Deviation of the Error:')
    print (np.std(errors))
    
    print ('Baseline: Root Mean Squared Error')
    print (math.sqrt(mean_squared_error(y_true, y_base)))
    print ('Mean_Absolute_Error:')
    errors=np.abs(y_true-y_base)
    print (np.mean(errors))
    print ('Standard Deviation of the Error:')
    print (np.std(errors))
    
    results=pd.DataFrame([y_true, y_pred_tree, rf_parameter, y_pred_svr, svm_parameter])
    results.to_csv(name+'regression_rf.csv')

    evaluate(y_pred_svr, y_pred_tree, y_true, y_base)
    
    return y_pred_svr, y_pred_tree, y_true, y_base, tree, svm_clf


# In[ ]:

def evaluate(y_pred_svr, y_pred_tree, y_true, y_base):

    tree_error=np.abs(y_true-y_pred_tree)
    svr_error=np.abs(y_true-y_pred_svr)
    base_error=np.abs(y_true-y_base)
    
    print ('SVR: ' + str(np.mean(svr_error)))
    print ('Tree: ' + str(np.mean(tree_error)))
    print ('base: ' + str(np.mean(base_error)))

    print ('Tree better then Baseline')
    print (stats.ttest_rel(tree_error, base_error, axis=0, nan_policy='omit'))
    
    print (stats.kruskal(tree_error, base_error))

    print ('SVR better then Baseline')
    print (stats.ttest_rel(svr_error, base_error, axis=0, nan_policy='omit'))
    
    print (stats.kruskal(svr_error, base_error))
    
    
def reg_cor(x, y, x_label, y_label):
    plt.figure(figsize=(12,8))
    sns.regplot(x, y)
    plt.yticks(fontsize=14)    
    plt.xticks(fontsize=14)  
    plt.xlabel(x_label, fontsize=16) 
    plt.ylabel(y_label, fontsize=16) 
    plt.savefig(x_label + y_label + '_Regression_Correlation.png')
    plt.show()
    plt.close()

