# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# importing libraries............

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#importing dataset................

data_ipl = pd.read_csv("C:\\Users\\Asus\\PycharmProjects\\untitled2\\dataset\\datasets_29843_429151_matches.csv")

#deleting unsignificant columns like umpire season..........


to_drop = ['season', 'date', 'dl_applied', 'umpire1', 'umpire2', 'umpire3', 'id', 'result', 'player_of_match']
data_ipl.drop(columns=to_drop, inplace=True)
#print(data_ipl.columns)
#print(data_ipl.head())


# cleaning data (merging delhi capitals and delhi daredevils and rising pune supergiant and rising pune super giants as they are same team)


data_ipl["team2"] = data_ipl["team2"].replace("Rising Pune Supergiant", "Rising Pune Supergiants")
data_ipl["team1"] = data_ipl["team1"].replace("Rising Pune Supergiant", "Rising Pune Supergiants")
data_ipl["winner"] = data_ipl["winner"].replace("Rising pune Supergiant", "Rising Pune Supergiants")
data_ipl["toss_winner"] = data_ipl["toss_winner"].replace("Rising Pune Supergiant", "Rising Pune Supergiants")
data_ipl["team2"] = data_ipl["team2"].replace("Delhi Daredevils", "Delhi Capitals")
data_ipl["team1"] = data_ipl["team1"].replace("Delhi Daredevils", "Delhi Capitals")
data_ipl["winner"] = data_ipl["winner"].replace("Delhi Daredevils", "Delhi Capitals")
data_ipl["toss_winner"] = data_ipl["toss_winner"].replace("Delhi Daredevils", "Delhi Capitals")

print(data_ipl.team1.unique())

# Filling the values of city based on venue
conditions = [data_ipl["venue"] == "Rajiv Gandhi International Stadium, Uppal",
              data_ipl["venue"] == "Maharashtra Cricket Association Stadium",
              data_ipl["venue"] == "Saurashtra Cricket Association Stadium",
              data_ipl["venue"] == "Holkar Cricket Stadium",
              data_ipl["venue"] == "M Chinnaswamy Stadium", data_ipl["venue"] == "Wankhede Stadium",
              data_ipl["venue"] == "Eden Gardens", data_ipl["venue"] == "Feroz Shah Kotla",
              data_ipl["venue"] == "Punjab Cricket Association IS Bindra Stadium, Mohali",
              data_ipl["venue"] == "Green Park",
              data_ipl["venue"] == "Punjab Cricket Association Stadium, Mohali",
              data_ipl["venue"] == "Dr DY Patil Sports Academy",
              data_ipl["venue"] == "Sawai Mansingh Stadium", data_ipl["venue"] == "MA Chidambaram Stadium, Chepauk",
              data_ipl["venue"] == "Newlands", data_ipl["venue"] == "St George's Park",
              data_ipl["venue"] == "Kingsmead", data_ipl["venue"] == "SuperSport Park",
              data_ipl["venue"] == "Buffalo Park", data_ipl["venue"] == "New Wanderers Stadium",
              data_ipl["venue"] == "De Beers Diamond Oval", data_ipl["venue"] == "OUTsurance Oval",
              data_ipl["venue"] == "Brabourne Stadium", data_ipl["venue"] == "Sardar Patel Stadium",
              data_ipl["venue"] == "Barabati Stadium",
              data_ipl["venue"] == "Vidarbha Cricket Association Stadium, Jamtha",
              data_ipl["venue"] == "Himachal Pradesh Cricket Association Stadium", data_ipl["venue"] == "Nehru Stadium",
              data_ipl["venue"] == "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
              data_ipl["venue"] == "Subrata Roy Sahara Stadium",
              data_ipl["venue"] == "Shaheed Veer Narayan Singh International Stadium",
              data_ipl["venue"] == "JSCA International Stadium Complex",
              data_ipl["venue"] == "Sheikh Zayed Stadium", data_ipl["venue"] == "Sharjah Cricket Stadium",
              data_ipl["venue"] == "Dubai International Cricket Stadium",
              data_ipl["venue"] == "M. A. Chidambaram Stadium",
              data_ipl["venue"] == "Feroz Shah Kotla Ground", data_ipl["venue"] == "M. Chinnaswamy Stadium",
              data_ipl["venue"] == "Rajiv Gandhi Intl. Cricket Stadium", data_ipl["venue"] == "IS Bindra Stadium",
              data_ipl["venue"] == "ACA-VDCA Stadium"]
values = ['Hyderabad', 'Mumbai', 'Rajkot', "Indore", "Bengaluru", "Mumbai", "Kolkata", "Delhi", "Mohali", "Kanpur",
          "Mohali", "Pune", "Jaipur", "Chennai", "Cape Town", "Port Elizabeth", "Durban",
          "Centurion", 'Eastern Cape', 'Johannesburg', 'Northern Cape', 'Bloemfontein', 'Mumbai', 'Ahmedabad',
          'Cuttack', 'Jamtha', 'Dharamshala', 'Chennai', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi',
          'Abu Dhabi', 'Sharjah', 'Dubai', 'Chennai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Mohali', 'Visakhapatnam']
data_ipl['city'] = np.where(data_ipl['city'].isnull(),
                            np.select(conditions, values),
                            data_ipl['city'])

# Removing records having null values in "winner" column


data_ipl = data_ipl[data_ipl["winner"].notna()]
data_ipl = data_ipl.reset_index(drop=True)


#encoding the numeric values

encoder= LabelEncoder()
data_ipl["team1"]=encoder.fit_transform(data_ipl["team1"])
data_ipl["team2"]=encoder.fit_transform(data_ipl["team2"])
data_ipl["winner"]=encoder.fit_transform(data_ipl["winner"].astype(str))
data_ipl["toss_winner"]=encoder.fit_transform(data_ipl["toss_winner"])
data_ipl["venue"]=encoder.fit_transform(data_ipl["venue"])
#outcome variable team1_win as a probability of team1 winning the match
data_ipl.loc[data_ipl["winner"]==data_ipl["team1"],"team1_win"]=1
data_ipl.loc[data_ipl["winner"]!=data_ipl["team1"],"team1_win"]=0

#outcome variable team1_toss_win as a value of team1 winning the toss
data_ipl.loc[data_ipl["toss_winner"]==data_ipl["team1"],"team1_toss_win"]=1
data_ipl.loc[data_ipl["toss_winner"]!=data_ipl["team1"],"team1_toss_win"]=0

#outcome variable team1_bat to depict if team1 bats first
data_ipl["team1_bat"]=0
data_ipl.loc[(data_ipl["team1_toss_win"]==1) & (data_ipl["toss_decision"]=="bat"),"team1_bat"]=1
data_ipl.drop(['city'],axis=1,inplace=True)
data_ipl.drop(['team1_bat','toss_decision'],axis=1,inplace=True)
X = data_ipl.iloc[:, [0,1,2,6,8]].values
y = data_ipl.iloc[:, 7].values


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0,shuffle=True)

#scaling data for better results.......

sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:,:] )
X_test[:, :] = sc.transform(X_test[:,:] )


#analysing different models


#Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of Logistic Regression Classifier on test set: {:.4f}'.format(logreg.score(X_test, y_test)))

#Decision Tree Classifier

dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)
print('Accuracy of Decision Tree Classifier on test set: {:.4f}'.format(dtree.score(X_test, y_test)))

#SVM


svm=SVC()
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
print('Accuracy of SVM Classifier on test set: {:.4f}'.format(svm.score(X_test, y_test)))

#Random Forest Classifier

randomForest= RandomForestClassifier(n_estimators=100)
randomForest.fit(X_train,y_train)
y_pred = randomForest.predict(X_test)
print('Accuracy of Random Forest Classifier on test set: {:.4f}'.format(randomForest.score(X_test, y_test)))



