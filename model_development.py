#Importing Dependencies
import pandas as pd
import numpy as np
import sklearn
from datetime import datetime
import pickle

#Importing Dataset
df = pd.read_csv('ipl.csv')

########################## Data Cleaning ###########################

#Removing unwanted columns
columns_remove = ['mid','venue','batsman','bowler','striker','non-striker']
df.drop(labels=columns_remove, axis=1, inplace=True)

#Keeping only consistent teams
consistent_teams = ['Kolkata Knight Riders','Chennai Super Kings','Rajasthan Royals',
                    'Mumbai Indians','Kings XI Punjab','Royal Challengers Bangalore',
                    'Delhi Daredevils','Sunrisers Hyderabad']

df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]

#Removing the first 5 overs data in every match
df = df[df['overs']>=5.0]

#Converting the column 'date' from string into datetime object
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

############################# Data Preprocessing #######################

#Converting to categorical features using OneHotEncoding method
df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

#Rearranging the columns
df = df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]

########################## Model Building ########################

#Splitting the data into train and test set
x_train = df.drop(labels='total', axis=1)[df['date'].dt.year <=2016]
x_test = df.drop(labels='total', axis=1)[df['date'].dt.year >=2017]

y_train = df[df['date'].dt.year <=2016]['total'].values
y_test = df[df['date'].dt.year >=2017]['total'].values

#Removing the 'date' column
x_train.drop(labels='date', axis=True, inplace=True)
x_test.drop(labels='date', axis=True, inplace=True)

########################## Model Implementation #########################
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

pred = regressor.predict(x_test)

#Creating a pickle file for the Model to generate Output
filename = 'IPL.pkl'
pickle.dump(regressor, open(filename, 'wb'))

######################### Model Evaluation ##############################
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import math

print("Mean squared error: %.2f" % mean_squared_error(y_test, pred))
print("R-Squared error: %.2f" % r2_score(y_test, pred))
print("Root mean squared error: %.2f" % math.sqrt(mean_squared_error(y_test, pred)))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, pred))
