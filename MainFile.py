#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install yfinance')


# In[2]:


import yfinance as yf


# In[3]:


sp500 = yf.Ticker("^GSPC")


# In[4]:


sp500 = sp500.history(period='max')


# In[5]:


sp500


# In[6]:


sp500.index


# In[7]:


# Show the trading days on the x-axis and show the Closing Price on the y-axis
sp500.plot.line(y='Close', use_index=True)


# In[8]:


# Removing the Dividends and Stock Splits columns from data
del sp500['Dividends']
del sp500['Stock Splits']


# In[9]:


# Trying to predict tomorrows price by taking the close column and shifting them back one day
sp500['Tomorrow'] = sp500['Close'].shift(-1)


# In[10]:


sp500


# In[11]:


# Based on tomorrows price we can set a target
# Use this for the ML
sp500['Target'] = (sp500['Tomorrow'] > sp500['Close']).astype(int)


# In[12]:


sp500


# In[13]:


# Removing all data that comes before 1990 (Might not be as useful for helping predict future prices)
# Need .copy() because if you don't use you'll get a pandas setting with copy warning when trying to subset a dataframe and 
# later assign back to it
sp500 = sp500.loc['1990-01-01':].copy()


# In[14]:


sp500


# In[15]:


pip install --upgrade scipy


# In[16]:


from sklearn.ensemble import RandomForestClassifier

# n_estimators is the number of individual decision trees we want to train:
             # the higher this is, the better the accuracy is up to a limit
# min_sample_split protects from over-fitting, the higher the less accurate but less it will over-fit
# setting a random_state means if we run the same model multiple times, the random numbers that are generated will be 
# in a predicable sequence each time using this random seed of one
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state = 1)


# Train set:
# Time series data shouldn't use cross validation
# We want the model to learn to predict
# Putting all the rows execpt the last 100 in train set and the last 100 in test set
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# This is going to train the model using the predictor columns and have it try and predict the target
predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
model.fit(train[predictors], train['Target'])


# In[17]:


from sklearn.metrics import precision_score

# Generate predictions using the model
preds = model.predict(test[predictors])


# In[18]:


# Turn the predictions into a Pandas series from a Numpy array
# Use same index as the test dataset
import pandas as pd
preds = pd.Series(preds, index=test.index)


# In[19]:


# Checking the precision score of the model
precision_score(test["Target"], preds)


# In[20]:


# Combine the actual values with the predictive values
# Axis=1 means treat each of these inputs as a column in the dataset
combined = pd.concat([test['Target'], preds], axis=1)


# In[21]:


# The orange line is our predictions and the blue is what actually happened
combined.plot()


# In[22]:


# Prediction function will wrap everthing above into one function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined


# In[23]:


# Start is the how much data you start the model with 
# Step is how much you should train the model
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    
    # Creating the training and test set
    # Training is prior years and test is the current year
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# In[24]:


predictions = backtest(sp500, model, predictors)


# In[25]:


# 0 means model predicts market goes down
# 1 means model predicts market goes up
predictions["Predictions"].value_counts()


# In[26]:


precision_score(predictions['Target'], predictions["Predictions"])


# In[27]:


# The percentage of days the market actually went up
# Value counts of the target / the number of rows total
predictions['Target'].value_counts() / predictions.shape[0]


# In[28]:


# checking on last 2 trading days, 5 trading days, etc.
horizons = [2,5,60,250,1000]
new_predictors = []

# loop through horizons and calculate a rolling average against that horizon
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    # Creating ratio_column and adding to sp500 dataframe
    ratio_column = f"Close_ratio_{horizon}"
    # The close price in the sp500(Todays Close) / the rolling average of (2 days, 5 days, etc)
    sp500[ratio_column] = sp500['Close'] / rolling_averages['Close']
    
    # Creating trend_column and adding to sp500 dataframe
    trend_column = f"Trend_{horizon}"
    # On any given day it will look at the previous days and find the sum of the target
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()['Target']
    
    new_predictors += [ratio_column, trend_column]


# In[29]:


# drop all the rows with NaN in the columns
sp500 = sp500.dropna()


# In[30]:


sp500


# In[31]:


model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# In[32]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    # This will return the probability of whether the stock price will go up
    preds = model.predict_proba(test[predictors])[:,1]
    # If the probabilty of the stock going up is 60% or greater, predict it to go up
    # This reduces total number of trading days, but increases the chance the price goes up on those days
    # Makes the model be more confident
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined


# In[33]:


predictions = backtest(sp500, model, new_predictors)


# In[34]:


predictions['Predictions'].value_counts()


# In[35]:


precision_score(predictions['Target'], predictions['Predictions'])


# In[36]:


# Summary of this 
    # Downloaded Stock Data
    # Cleaned and Visualized the data
    # Setup ML Target 
    # Trained Initial model
    # Evaluted error and created a way to backtest and accurately measured over long periods of time
    # Improved model with extra predictor columns

# Ways to extend this model
    # Some exchanges are open overnight: Looking at those prices and correlate them
    # Add in news
    # Adding in key components of the SP500: Key stock, sectors
    # Increasing the resolution: try looking at hourly, minute by minute data to get more accurate predictions

