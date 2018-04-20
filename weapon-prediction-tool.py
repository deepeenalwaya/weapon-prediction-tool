
#%% LIBRARY IMPORTS

import numpy as np
import pandas as pd
from pandas import ExcelWriter
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from sklearn.model_selection import train_test_split

#%% DATA IMPORTS

df1 = pd.read_excel('fatal-police-preprocessed-revised.xlsx')
df2 = pd.read_excel('unknown_weapon.xlsx')

#%% DATA ANALYSIS

# Category Counts

armed_count = len(set(df1['armed']))
gender_count = len(set(df1['gender']))
race_count = len(set(df1['race']))
city_count = len(set(df1['city']))
mental_count = len(set(df1['signs_of_mental_illness']))
threat_level_count =  len(set(df1['threat_level']))
flee_count =  len(set(df1['flee']))
body_camera_count =  len(set(df1['body_camera']))
state_count = len(set(df1['state']))
state_bucket_count = len(set(df1['state_bucket']))
age_count = len(set(df1['age']))
age_bucket_count = len(set(df1['age_bucket']))

# Data Dimensions

df1_rows = df1.shape[0]
df1_columns = df1.shape[1]

#%% NEURAL NETWORK - BUILDING

# Model Parameters

default_parameters = input('Build model with default parameters? Type yes or no')

if default_parameters in ['yes']:
    test_data_fraction = 0.25
    train_batch_size = 10
    train_num_epochs = 1000
    train_num_steps = 1000
    hidden_layer_units = [2,2]
    
if default_parameters in ['no']:
    test_data_fraction = float(input('Fraction of test data:'))
    train_batch_size = int(input('Batch size for training:'))
    train_num_epochs = int(input('Number of training epochs:'))
    
    num_of_layers = int(input('Enter number of layers:'))
    
    hidden_layer_units = list()
    
    for i in range (1,num_of_layers+1):    
        print('For layer ',i,':')
        current_layer_neurons = int(input('No of neurons in layer:'))
        hidden_layer_units.append(current_layer_neurons)
        
    train_num_steps = int(input('Number of training steps:'))


# Normalizing the numeric features

cols_to_norm = ['age']
df1[cols_to_norm] = df1[cols_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min()))

# Defining the numeric features

age = tf.feature_column.numeric_column('age')

# Defining the categorical features

gender = tf.feature_column.categorical_column_with_vocabulary_list('gender',['M','F'])
race = tf.feature_column.categorical_column_with_hash_bucket('race',hash_bucket_size=race_count)
mental = tf.feature_column.categorical_column_with_vocabulary_list('signs_of_mental_illness',['Yes','No'])
threat_level = tf.feature_column.categorical_column_with_hash_bucket('threat_level',hash_bucket_size=threat_level_count)
flee = tf.feature_column.categorical_column_with_hash_bucket('flee',hash_bucket_size=flee_count)
body_camera = tf.feature_column.categorical_column_with_vocabulary_list('body_camera',['Yes','No'])
state = tf.feature_column.categorical_column_with_hash_bucket('state',hash_bucket_size=state_count)
state_bucket = tf.feature_column.categorical_column_with_hash_bucket('state_bucket',hash_bucket_size=state_bucket_count)
age_bucket = tf.feature_column.categorical_column_with_hash_bucket('age_bucket',hash_bucket_size=age_bucket_count)

# Embedding the categorical features

embedded_gender_col = tf.feature_column.embedding_column(gender,dimension=gender_count)
embedded_race_col = tf.feature_column.embedding_column(race,dimension=race_count)
embedded_mental_col = tf.feature_column.embedding_column(mental,dimension=mental_count)
embedded_threat_level_col = tf.feature_column.embedding_column(threat_level,dimension=threat_level_count)
embedded_flee_col = tf.feature_column.embedding_column(flee,dimension=flee_count)
embedded_body_camera_col = tf.feature_column.embedding_column(body_camera,dimension=body_camera_count)
embedded_state_col = tf.feature_column.embedding_column(state,dimension=state_count)
embedded_state_bucket_col = tf.feature_column.embedding_column(state_bucket,dimension=state_bucket_count)
embedded_age_bucket_col = tf.feature_column.embedding_column(age_bucket,dimension=age_bucket_count)


# Choosing the inputs and defining the feature columns

feat_cols = []
features_data_list = []

default_inputs = input('Use the default input columns? Enter yes or no')

if default_inputs in ['yes']:
    
    feat_cols = [age,embedded_gender_col,embedded_state_col,embedded_race_col,embedded_mental_col,embedded_threat_level_col,embedded_flee_col,embedded_body_camera_col]
    features_data = df1[['age','gender','state','race','signs_of_mental_illness','threat_level','flee','body_camera']]
    features_data_2 = df2[['age','gender','state','race','signs_of_mental_illness','threat_level','flee','body_camera']]
    
if default_inputs in ['no']:
    
    feat_cols = [age,embedded_gender_col,embedded_state_col,embedded_race_col,embedded_mental_col,embedded_threat_level_col,embedded_flee_col,embedded_body_camera_col,embedded_state_bucket_col]
    features_data = df1[['age','gender','state','race','signs_of_mental_illness','threat_level','flee','body_camera','state_bucket']]
    features_data_2 = df2[['age','gender','state','race','signs_of_mental_illness','threat_level','flee','body_camera','state_bucket']]
    
    input_dict = {0:'age',1:'gender',2:'state',3:'race',4:'signs_of_mental_illness',5:'threat_level',6:'flee',7:'body_camera',8:'state_bucket'}
    input_data_dict = {0:age,1:embedded_gender_col,2:embedded_state_col,3:embedded_race_col,4:embedded_mental_col,5:embedded_threat_level_col,6:embedded_flee_col,7:embedded_body_camera_col,8:embedded_state_bucket_col}
    
    print('Following inputs are available: \n')
    
    for i in input_dict:
        print(i,input_dict[i])
    
    multiple_unwanted_inputs = []
    
    for remove_iter in range(0,100):
        
        remove_input = input('Do you want to remove an input?')
        
        if remove_input in ['yes']:
            
            
            unwanted_input = int(input('Enter the number of the input you want to remove:'))
            del features_data[input_dict[unwanted_input]]
            del features_data_2[input_dict[unwanted_input]]
            multiple_unwanted_inputs.append(unwanted_input)
            
        if remove_input in ['no']:
            break
    
    multiple_unwanted_inputs.sort()
    
    for each_unwanted in multiple_unwanted_inputs:
        
        feat_cols[each_unwanted]='need to delete this'
        
    for each in range(0,feat_cols.count('need to delete this')):
        feat_cols.remove('need to delete this')
       
# Defining the output data 

labels_data = df1['armed']

# Building the model

x_train, x_test, y_train, y_test = train_test_split(features_data,labels_data,test_size=test_data_fraction)
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=train_batch_size,num_epochs=train_num_epochs,shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=hidden_layer_units,feature_columns=feat_cols,n_classes=4)

#%% NEURAL NETWORK - TRAINING

dnn_model.train(input_fn = input_func, steps=train_num_steps)

#%% NEURAL NETWORK - TESTING

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test,y=y_test,batch_size=10, num_epochs=1, shuffle=False)
results = dnn_model.evaluate(eval_input_func)
print(results['accuracy'])

#%% NEURAL NETWORK - PREDICTIONS

#x_pred = features_data_2
#pred_input_func = tf.estimator.inputs.pandas_input_fn(x=x_pred,batch_size=10,num_epochs=1,shuffle=False)
#
#predictions = dnn_model.predict(pred_input_func)
#
#for x in list(predictions):
#    print(x['class_ids'][0])

