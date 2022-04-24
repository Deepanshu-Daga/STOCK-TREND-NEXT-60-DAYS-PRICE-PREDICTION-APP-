from operator import mod
from re import L
from time import time
from turtle import width
import numpy as np 
import math
import pandas as pd 
import pandas_datareader as pdr 
import matplotlib.pyplot as plt
from pyparsing import col
from sklearn.preprocessing import MinMaxScaler
from keras.models import sequential
from keras.layers import Dense , LSTM
from keras.models import Sequential
from torch import inner
plt.style.use("fivethirtyeight")
import datetime
from datetime import  timedelta
from keras.models import load_model                             # not going to run epoches everytime thus to run models

import streamlit as st


# Get the stock quote
# name should be exact from yahoo finance site

# creating a header of webpage
st.title('Deepanshu Daga  :Deep Learning Stock Trend Prediction Platform')


# askimg for a  user input   and by default it will be HDFCBANK.NS
stock_name= st.text_input('Enter the Stock Ticker' , 'BANDHANBNK.NS')

st.write('Note Use These Only : ntpc.ns','heromotoco.ns','techm.ns','coalindia.ns','apollohosp.ns','itc.ns','hindalco.ns','lti.ns','indusindbk.ns','amarajabat.ns','bergepaint.ns','gujgasltd.ns','hdfc.ns','hdfcamc.ns','hindunilvr.ns','igl.ns','irfc.ns','petronet.ns','bandhanbnk.ns')

# Interactive stock select user_input
df = pdr.DataReader( stock_name , data_source='yahoo',start='2000-01-01',end=datetime.datetime.now())


# Describing the data 
st.subheader('Data from 2000 - Present Date')
st.write(df.describe())


# visualization 
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,8))
plt.plot(df.Close , linewidth=2)
st.pyplot(fig)




# visualization of 100 and 200 moving avg
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig = plt.figure(figsize=(12,8))
plt.plot(ma200 ,'b', linewidth=2)
plt.plot(ma100  ,'r', linewidth=2)
plt.plot(df.Close , 'g',linewidth=1)
plt.legend(['100MA' , '200MA' , 'Close Price'] , loc = 'lower right')
st.pyplot(fig)


# Creating a new df for only close price 
#data = df.filter(['Close','High'])           # gives 2 column 
data = df.filter(['Close'])                   # gives 1 column
#data

# create df to a  num py array         df.values : Only the values in the DataFrame will be returned, the axes labels will be removed.

dataset = data.values
#dataset


# Get the no of adta to train the model on say 80%
training_data_len = math.ceil(len(dataset)*0.80)                    #Return the ceiling of x as an Integral.
#training_data_len

# scale the data 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)                        # range [0-1] on numpy array of dataset
#scaled_data



# create a scaled training data set 
train_data = scaled_data[0:training_data_len , : ]                # till row = training data len all column points

# split the data into x_train  , and y_train data sets
x_train = []                                                      # features = independent variables 
y_train = []                                                      # lables = output = dependent variables 

for i in range(60 , len(train_data)): 
    x_train.append(train_data[i-60:i , 0])                        # we are going to append 0 to 59th  values of train_data in 0th column
    y_train.append(train_data[i , 0])                             # lable/ prediction is 60th data point of 0th column from train_data to y_train
    
   
# # no need to train model now just load it
model = load_model('{}_keras_model.h5'.format(stock_name))     


# creating the testing data set
# create a new array containing scaled values from index  end of len(train_data) till end of the total data

test_data = scaled_data[training_data_len-60 : , : ]         # all data from 60th to end of data with all column

# creating a dataset x_test and y_test
x_test = []
y_test = dataset[training_data_len : , : ]                          # actual prediction values
                                                                    # dataset values are original closing prce and not scaled

for i in range(60 , len(test_data)):
    x_test.append(test_data[i-60 : i, 0 ])
    


# create the data to a numpy array as model need 3d array
x_test = np.array(x_test)

# reshape the data for the model      (newshape =no of rows = x_test.shape[0], timesteps = 60 days = x_test.shape[1] , and 1 )
x_test = np.reshape(x_test , newshape= (x_test.shape[0] , x_test.shape[1] , 1 ))


y_predictions = model.predict(x_test)    # scaled value




# undo scaling
                                                           # we want the predictions to be exact same as real lables 
y_predictions = scaler.inverse_transform(y_predictions)        # Undo the scaling of X according to feature_range(0-1)
                                                           # gaining back the real value

# EVALUATE THE MODEL root mean square error (RMSE)
# its a measure of how good the model predicts the response 
# can be compared by std deviation less
# lower value of rmse is better : try diff range of dataset 

rmse = np.sqrt(np.mean(y_predictions - y_test)**2)
st.subheader('Root mean square error in predicted vs actual values')
st.write(rmse)
    
    


# plot the data
train = data[ : training_data_len]                              # train = x_train
valid = data[training_data_len : ]                              # valid = y_test                
valid['predictions'] = y_predictions                       # valid(prediction) = y_prediction

# visualise the model 
st.subheader('Predicted Vs Original')
fig_2=plt.figure(figsize=(20,12))
plt.title('Model')
plt.xlabel('Date' , fontsize = 18)
plt.ylabel('closing price INR' , fontsize = 18)
plt.plot(train['Close'] , linewidth=2)
plt.plot(valid[['Close', 'predictions']] , linewidth=2)
plt.legend(['TRAIN' , 'Original' , 'PREDICTION'] , loc= 'lower right')
plt.show()
st.pyplot(fig_2)





# Prediction For Next 60 Days




model = load_model('{}_keras_model.h5'.format(stock_name))
# Get the quotes
stock_quote = pdr.DataReader( stock_name ,data_source='yahoo',start='2000-01-01',end=datetime.datetime.now())
# create a new dataframe 
new_df = stock_quote.filter(['Close'])
# Get the last 60days closing price value and convert the data form tp numpy array
last_60days = new_df[-60 :].values
# Scale the data as per previous tramsformation made
last_60days_scaled = scaler.transform(last_60days)     # here not using fit.transform because we want to transform by using same last value ranges thus not creating new one
# create an empty list 
X_TEST = []
# Append last 60 days 
X_TEST.append(last_60days_scaled)
#print(X_TEST)
# convert the X_TEST  to a numpy array
X_TEST = np.array(X_TEST)
# Reshape the data for LSTM model 
X_TEST = np.reshape(X_TEST , newshape= (X_TEST.shape[0] ,X_TEST.shape[1] , 1 ))   # last is 1 because 1 feature of close price is required
#print(X_TEST.shape)
# loading the model
pred_price = model.predict(X_TEST)


st.subheader('Future Predicted Value of Tommorrow : ')
tomorrow_price = scaler.inverse_transform((pred_price))
st.write(tomorrow_price)





unscaled_pred_price = []
for i in range(0,60):
    
    
    pred_price = model.predict(X_TEST)
    unscaled_pred_price.append(scaler.inverse_transform((pred_price)))
    # Get the predicted scaled price 



   
    last_60days_scaled = np.append(last_60days_scaled , float(pred_price) )
    last_60days_scaled = last_60days_scaled[-60 :]
    X_TEST = []
    X_TEST.append(last_60days_scaled)
    X_TEST = np.array(X_TEST)
    X_TEST = np.reshape(X_TEST , newshape= (X_TEST.shape[0] ,X_TEST.shape[1] , 1 ))   # last is 1 because 1 feature of close price is required



    
st.subheader('Prediction For Next 60 Days')  

from datetime import date , timedelta   
current_date = date.today()
# adding the predicted price and dates dataframe together
lst_pred_date = []
for i in range(0,60):
    pred_date = (date.today() + timedelta(i)).isoformat()
    lst_pred_date.append(pred_date)


# currently pred_date is in string list form thus can not be concatenated thus converting to dataframe.

# df_pred_date = pd.DataFrame(lst_pred_date)  
                   
unscaled_pred_price= np.array(unscaled_pred_price, dtype=np.float32)       # its (60,1,1) shape  array 3d array
#                                                                            # converting back to 1 d array so that we can mearge it with future date array
unscaled_pred_price=unscaled_pred_price.reshape(60,)
df_unscaled_pred_price = pd.DataFrame(unscaled_pred_price)

# renaming column names
# df_unscaled_pred_price = df_unscaled_pred_price.rename(columns={'0' : 'a'}, inplace=True)
# df_pred_date = df_pred_date.rename(columns={'0' : 'b'}, inplace=True)




st.write(unscaled_pred_price.shape)                            # checking for shape 
#st.write(pd.concat([df_unscaled_pred_price , df_pred_date ] , axis=1 , join='inner'))

st.write(df_unscaled_pred_price)
# st.write(df_pred_date) 