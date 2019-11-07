import pandas as pd
import numpy as np
import scipy

# load dataframe

def load(study):
    
    if study=='mb':
        path='./mb/mb_EMG.csv'
    if study=='charite':
        path='./charite_OpenFace_205_features.csv'
        audio_path='./charite_audio_final.csv'

    #Audio
    df_audio=pd.read_csv(audio_path, sep=',', na_values=['?'])
    #df_audio['frame']=df_audio.counter
        
    #Face
    df=pd.read_csv(path, sep=',', na_values=['?'])
    df.vpn=df.vpn.str[1:3].astype(int)
    
    # exclude due to exclusion criteria 
    df_audio=df_audio[df_audio.vpn!=54].reset_index(drop=True)
    df_audio=df_audio[df_audio.vpn!=38].reset_index(drop=True)
    df_audio=df_audio[df_audio.vpn!=13].reset_index(drop=True)
    df_audio=df_audio[df_audio.vpn!=1].reset_index(drop=True)
    
        # exclude due to exclusion criteria 
    df=df[df.vpn!=54].reset_index(drop=True)
    df=df[df.vpn!=38].reset_index(drop=True)
    df=df[df.vpn!=13].reset_index(drop=True)
        
        
    action_r=['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 
             'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 
             'AU25_r', 'AU26_r', 'AU45_r']
    
    action_c=['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 
              'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 
              'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

    gaze=['gaze_angle_x', 'gaze_angle_y']
       
    df_audio=df_audio.rename(columns={'0' : 'spectrum_0', '1': 'spectrum_1', '2': 'spectrum_2', 
                          '3' : 'spectrum_3', '4': 'spectrum_4', '5': 'spectrum_5',
                          '6' : 'spectrum_6', '7': 'spectrum_7', '8': 'spectrum_8',
                          '9' : 'spectrum_9', '10': 'spectrum_10', '11': 'spectrum_11',
                          '12' : 'spectrum_12', '13': 'spectrum_13', '14': 'spectrum_14',
                          '15' : 'spectrum_15', '16': 'spectrum_16', '17': 'spectrum_17',
                          '18' : 'spectrum_18', '19': 'spectrum_19', '20': 'spectrum_20',
                          '21' : 'spectrum_21', '22': 'spectrum_22', '23': 'spectrum_23',
                          '24': 'spectrum_24', '25' : 'spectrum_25', '26': 'spectrum_26', 
                          '27': 'spectrum_27', '28' : 'spectrum_28', '29': 'spectrum_29',
                          '30': 'spectrum_30', '31' : 'spectrum_31', '32': 'spectrum_32', 
                          '33': 'spectrum_33',  '34' : 'spectrum_34', '35': 'spectrum_35',
                          '36': 'spectrum_36',  '37' : 'spectrum_37', '38': 'spectrum_38', 
                          '39': 'spectrum_39'                          
              })

    audio=['pitch','meanF0Hz', 'stdevF0Hz', 'HNR',
           'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter',
           'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer',
           'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer', 'JitterPCA', 'ShimmerPCA',
           'spectrum_0', 'spectrum_1', 'spectrum_2', 'spectrum_4', 'spectrum_3', 
             'spectrum_5', 'spectrum_6', 'spectrum_7', 'spectrum_8', 'spectrum_9',
             'spectrum_10', 'spectrum_11', 'spectrum_12',  'spectrum_13', 'spectrum_14',
             'spectrum_15', 'spectrum_16', 'spectrum_17', 'spectrum_18', 'spectrum_19',
             'spectrum_20', 'spectrum_21',  'spectrum_22', 'spectrum_23', 'spectrum_24', 
             'spectrum_25', 'spectrum_26', 'spectrum_27', 'spectrum_28', 'spectrum_29', 
             'spectrum_30', 'spectrum_31', 'spectrum_32', 'spectrum_33', 'spectrum_34',
             'spectrum_35', 'spectrum_36', 'spectrum_37', 'spectrum_38', 'spectrum_39']
    

    
    # load dataframe
    df=df.rename(columns={'disgust_proband': 'disgust_participant', 
                      'neutral_proband' : 'neutral_participant', 
                          'joy_proband': 'joy_participant'
                      })



    df['timepoint']=df.frame/30
    #df['conversationnr']=df.conversation
    di = {"intro":1, 
          "neutral_speaker":2,
          "neutral_participant":3,
          "joy_speaker":4, 
          "joy_participant":5, 
          "disgust_speaker":6, 
          "disgust_participant":7}
    #df=df.replace({"conversationnr": di})

    
    #to make the sequences comparable in length as there is around one second variation

    return df, df_audio, action_r, action_c, gaze, audio

def load_experts(df):
    path='./Ratings.csv'
    experts=pd.read_csv(path, sep=';', na_values=['?'], decimal=',')
    #experts['rater']=experts['Unnamed: 0']
    experts.columns

    experts=pd.merge(df.groupby('vpn').mean().reset_index(), experts, on='vpn')

    
    columns=['vpn', 'Pseudo', 'Prerating', 'Scaling', 'ASC', 'Value', 'rater',
             'experience_month_total', 'experience_month_diagnostic',
           'experience_diagnosis_cases', 'experience_ados', 'asc', 
            'asq', 'asd', 'sex', 'age', 'main_diagnosis', 'ados_commu',
       'ados_social', 'ados_total', 'adir_social', 'adir_commu', 'adir_behav',
       'adir_onset']
    
    #experts.ASC is the diagnostic assumption of the rater
    
    experts=experts[columns].reset_index(drop=True)
    
 #   experts['correct']=experts['asc']==experts['ASC']
 #   experts['false']=(experts['correct']==False)
 #   experts['false_positive']=(experts['correct']==False)&(experts['asc']==0)
 #   experts['false_negative']=(experts['correct']==False)&(experts['asc']==1)

  #  experts['true_negatives']=(experts['correct']==True)&(experts['asc']==0)
  #  experts['true_positives']=(experts['correct']==True)&(experts['asc']==1)
   
    return experts

def aq_charite(df):
    path='./charite/asq.csv'

    aq=pd.read_csv(path, sep=';', na_values=['-99']) 
    aq = aq.rename(columns={'Probanden-ID': 'vpn', 'ASQ': 'asq'})
    aq.vpn=aq.vpn.astype(str)
    aq.vpn=aq.vpn.str[-9:-7]
    aq.vpn=aq.vpn.astype(str).astype(int)
    df.vpn=df.vpn.astype(int)
    final=pd.merge(df, aq, on='vpn')

    return final



def ados_charite(df):
    path='./charite/ados.csv'
    ados=pd.read_csv(path, sep=';', na_values=['-99']) 
    ados=ados.rename(columns={'id': 'vpn'})
    ados.vpn=ados.vpn.astype(str)
    ados.vpn=ados.vpn.str[-9:-7]
    ados.vpn=ados.vpn.astype(str).astype(int)
    df.vpn=df.vpn.astype(int)
    final=pd.merge(df, ados, on='vpn')
    final['asc']=(final.asd==11)
    
    if np.mean(final[final.vpn==53].asc)==0:
        print ('Proband 53 ist korrekt als NT gelabelt')
    else:
        print ('Proband 53 ist falsch als ASC gelabelt')
        
        
    return final




# passiert bereits im Cutting!
def adapt_times(final):
    #final=final[(final.timestamp>final.conversation_start)]

    #categorical encoding
    dummies=final[['intro', 'neutral_speaker', 'neutral_participant', 'joy_participant',
               'joy_speaker', 'disgust_speaker', 'disgust_participant']]


    final["conversation"] = dummies.idxmax(axis=1)    
    #include a counter of Frames from the Begininng of the Interaction
    final['counter']=final.groupby(['vpn', 'conversation']).cumcount() 
    
    return final



#Hiermit lassen sich die Videos pruefen  
def check_videos_manual(df):
    conversation_parts=['intro', 'neutral_speaker', 'neutral_participant', 'joy_participant',
               'joy_speaker', 'disgust_speaker', 'disgust_participant']

    expected_times=['183s', '40s' ,'26s', '29s', 
                        '26s', '29s', '26s']
    for i in conversation_parts:
        print (i)
        print (df[df[i]==True].groupby('vpn').min()[['timestamp', 'frame']])
        print (df[df[i]==True].groupby('vpn').max()[['timestamp', 'frame']])

# allgemeine Checks des Dataframes
def check(df):
    for i in set(df.vpn):

        vpn_df=df[df.vpn==i].reset_index(drop=True)

        
        if len(vpn_df)!=11339: #11339
            if len(vpn_df)!=11309: #check(df)check(df)
                print ('Achtung: Falsche Zeilenanzahl bei Proband ' + str(i) )
                print (len(vpn_df))
            # die Gesamte Unterhaltung dauert sechseinhalb Minuten und ist aufgenommen mit 30 frames/sec 

        if max(vpn_df.counter)!=5638:
            print ('Achtung: Falscher Counter bei Proband ' + str(i) )
            print (max(vpn_df.counter))
            5638
            # Counter counts frames of the conversation (as 'frames', but starting with 0 not 2)

        if np.round(max(vpn_df.timepoint))!=378:
            if np.round(max(vpn_df.timepoint))!=377:
                print ('Achtung: Falscher Timepoint bei Proband ' + str(i) )
                print (np.round(max(vpn_df.timepoint)))
            # Timepoint shows time of the conversation (starting from 0.33 for first frame)

        if (vpn_df.start[0]>vpn_df.timer_neutral_speaker_start[0]) | (vpn_df.timer_neutral_speaker_start[0]>vpn_df.timer_neutral_proband_start[0]):
            print ('Fehler in Timing of Neutral-Part')

        if (vpn_df.timer_neutral_proband_end[0]>vpn_df.timer_joy_speaker_start[0]) | (vpn_df.timer_joy_speaker_start[0]>vpn_df.timer_joy_proband_start[0]):
            print ('Fehler in Timing of Joy-Part')

        if (vpn_df.timer_joy_proband_end[0]>vpn_df.timer_disgust_speaker_start[0]) | (vpn_df.timer_disgust_speaker_start[0]>vpn_df.end[0]):
            print ('Fehler in Timing of Disgust-Part')
            
    print ('alle Zeiten in Ordnung')
   
    if np.mean(df[df.vpn==53].asc)==0:
        print ('Proband 53 ist korrekt als NT gelabelt')
        
def load_VIT_actress_AU():
    VIT=pd.read_csv('./charite/actress_OpenFace_2.0.0_features.csv', sep=',', na_values=['?'])

    di = {"'intro-mit-atempausen.csv'": "intro", 
          "'neutral_proband.csv'": "Pneutral",
          "'neutral_sprecher.csv'":"Sneutral",
          "'joy_proband.csv'":"Pjoy", 
          "'joy_sprecher.csv'":"Sjoy", 
          "'disgust_proband.csv'":"Pdisgust", 
          "'disgust_speaker_new.csv'": "Sdisgust"}

    VIT=VIT.replace(di)

    VIT.rename(columns=lambda x: x.replace(" ", ""), inplace=True) 
    
    VIT['part']=VIT.vpn.str[1:]
    VIT['who']=VIT.vpn.str[0:1]
    
    di = {"intro":1, 
          "Sneutral":2,
          "Pneutral":3,
          "Sjoy":4, 
          "Pjoy":5, 
          "Sdisgust":6, 
          "Pdisgust":7}

    VIT=VIT.replace({"vpn": di})
    VIT=VIT.rename(columns={"vpn": "conversation_part"})
    
    VIT=VIT.sort_values(by=['conversation_part', 'frame']).reset_index(drop=True)
    VIT['counter']=VIT.index
    VIT['timepoint']=VIT.counter/25
    VIT['vpn']=0


    return VIT


def exclude_outlier(df):

    df['success_rate']=df['success'].groupby(df['vpn']).transform('mean') 
    print (set(df[df['success_rate']<0.9].vpn))
    print ('all participants excluded that were tracked with a successrate less than 0.9')
    df=df[df['success_rate']>0.9].reset_index(drop=True)

    print (len(df))
    df=df[df.success==1].reset_index(drop=True) #exclude non-sucessfull tracks
    print (len(df))
    print ('all non-successfully tracked frames excluded')

    df=df[df.confidence>0.75].reset_index(drop=True)
    print ('all frames excluded that were tracked with a lower confidence than 0.75')
    print (len(df))

    return df

def cut_dataframes(df):

    #df=df[df.frame<=11310].reset_index(drop=True)
        
    df['duration_total']=df['frame'].groupby(df['vpn']).transform('max') 
    print ('participants with a lower duration than 11309:')
    print (set(df[df['duration_total']<11309].vpn))
    df=df[df['duration_total']>11309].reset_index(drop=True)
    print ('were excluded!')
    
    return df

### exclude badly tracked participants
def exclusion_function(df):
    threshold=df.groupby('vpn').mean().confidence.mean()-df.groupby('vpn').mean().confidence.std()  
    df['vpn_conf_int']=df.groupby(['vpn', 'instruction']).confidence.transform('mean')
    df['exclusion']=(df['vpn_conf_int']<threshold)*1
    df=df.groupby('vpn').filter(lambda x: x['exclusion'].mean() == 0.)
    
    
def outlier_trials(df, var, outl):
    for v in var:
        
        mean=np.nanmean(df[v], axis=0)
        std=np.nanstd(df[v], axis=0)

        if outl=='sd':
            error=np.abs(df[v]-mean) # es ist unsinn die absoluten Werte der Differenz zu nehmen!!
            out_ind=(np.abs(df[v]-mean))>(3*std)

        if outl=='iq':
            quartile_1, quartile_3 = np.percentile(df[v], [25, 75])
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr * 3)
            upper_bound = quartile_3 + (iqr * 3)
            out_ind=(df[v] > upper_bound) | (df[v] < lower_bound)

        df.loc[out_ind, v]=np.nan  
        
        print (df[v].isna().sum())
            
        df.loc[:, v]=df[v].fillna(method='ffill')
        df.loc[:, v]=df[v].fillna(method='bfill')
        
    return df   

# gaze feature calculation
def calc_speed(df, var):

    epochs = pd.DataFrame()

    speeds = []
    # accelerations
    accs = []
    
    
    #values = epoch.loc[:, ['gaze_0_x', 'gaze_0_y', 'gaze_0_z']]
    #gaze_angle_x

    for i, ((vpn, conversation), epoch) in enumerate(df.groupby(['vpn', 'conversation'])):

  #  for i, ((vpn), epoch) in enumerate(df.groupby(['conversation', 'vpn'])):
        for v in var:
            values = epoch.loc[:, var]
            speed = np.diff(values[v], axis=0) #erste Ableitung
            acc = np.diff(speed, axis=0) #zweite Ableitung
            speed = np.linalg.norm(speed) #absolute Werte
            print (speed)
            acc = np.linalg.norm(acc)

            speeds.append(speed)
            accs.append(acc)

            epochs.loc[i, ('mean_speed_'+v)] = np.nanmean(speed)
            epochs.loc[i, ('mean_acc_'+v)] = np.nanmean(acc)
            epochs.loc[i, ('var_speed_'+v)] = np.nanvar(speed)
            epochs.loc[i, ('var_acc_'+v)] = np.nanvar(acc)

        epochs.loc[i, 'vpn'] = vpn
        epochs.loc[i, 'conversation'] = conversation
        #epochs.loc[i, 'asc'] = np.float(np.mean(epoch.loc[:, ['asc']]))
        #epochs.loc[i, 'frame'] = frame
        

    speeds = pd.DataFrame(speeds)
    accs = pd.DataFrame(accs)

    speeds = pd.concat([speeds, epochs], axis=1)
    accs = pd.concat([accs, epochs], axis=1)

    
    return epochs


def calc_gaze_var(df):
    
    df['gaze_angle_x']=df.gaze_angle_x*(-1)
    df['gaze_angle_y']=df.gaze_angle_y*(-1)

    center = lambda x: (x - x.median()) 
    df['gaze_angle_y_centered']=df.groupby(['vpn'])['gaze_angle_y'].transform(center)
    center = lambda x: (x - x.median()) 
    df['gaze_angle_x_centered']=df.groupby(['vpn'])['gaze_angle_x'].transform(center)

    df['gaze_angle_x_abs']=np.abs(df['gaze_angle_x_centered'])
    df['gaze_angle_y_abs']=np.abs(df['gaze_angle_y_centered'])
    
    df['gaze_angle_total_centered_abs']=df.gaze_angle_x_abs+df.gaze_angle_y_abs
    
    df['proband']=(df.conversation=='joy_proband')|(df.conversation=='ekel_proband')|(df.conversation=='neutral_proband')
    df['actress']=(df.conversation=='joy_speaker')|(df.conversation=='ekel_speaker')|(df.conversation=='neutral_speaker')
    df.loc[(df.conversation!='intro'), "whospeaks"] = df[['proband', 'actress']].idxmax(axis=1) 
    
    return df

#mimicry feature calculation

def calc_mim(df, VIT):

    def smooth(df, VIT=True):
        df.index = pd.to_datetime(df['timepoint'].astype('float64'), unit='s')
        if VIT==False:
            df=df.groupby(['vpn']).resample('10S').mean().reset_index(drop=True)
        if VIT==True:
            print ('error')
            df=df.resample('10S').mean().reset_index(drop=True)
        return df

    VIT=smooth(VIT.reset_index(drop=True), False)
    df=smooth(df.reset_index(drop=True), False)

    mimicry=pd.DataFrame()
    features=['AU06_r', 'AU12_r', 'AU06_c', 'AU12_c',
              'AU15_r', 'AU09_r', 'AU04_r',
              'AU15_c', 'AU09_c', 'AU04_c',
              'gaze_angle_x', 'gaze_angle_y']
   
    
    new_features=list()
    for feature in features:

        i=0
        print (feature)
        for vp in set(df.vpn):
    #        plt.scatter(df[df.vpn==vp][feature][1::], VIT[feature])
    #        plt.show()
            try:
                value, _=scipy.stats.spearmanr(df[df.vpn==vp][feature][1::], VIT[feature])
            except:
                value, _=scipy.stats.spearmanr(df[df.vpn==vp][feature], VIT[feature])
            mimicry.loc[i, 'vpn']=vp
            var_name='correlation_'+feature
            mimicry.loc[i, var_name]=value
            i=i+1
        new_features.append(var_name)
        
    return mimicry, new_features