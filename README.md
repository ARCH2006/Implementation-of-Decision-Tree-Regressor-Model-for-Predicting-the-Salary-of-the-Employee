# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
step 1.Import the required libraries. 
step 2.Read the data frame using pandas. 
step 3.Get the information regarding the null values present in the dataframe. 
step 4.Apply label encoder to the non-numerical column inoreder to convert into numerical values. 
step 5.Determine training and test data set. 
step 6.Apply decision tree regression on to the dataframe. 
step 7.Get the values of Mean square error, r2 and data prediction.
```
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ARCHANA S
RegisterNumber:212223040019
*/
import pandas as pd

data=pd.read_csv("C:/Users/SEC/Downloads/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:
```
head() for salary
```
![Screenshot 2024-10-10 150731](https://github.com/user-attachments/assets/c4920c18-f11d-41f5-b3d1-f1fc2dfaaae6)
```
MSE value:
```
![Screenshot 2024-10-10 150740](https://github.com/user-attachments/assets/6a48f875-0f3f-410a-ab91-0148645ce329)
```
R2 value:
```
![Screenshot 2024-10-10 150750](https://github.com/user-attachments/assets/6e508fa2-b21e-419b-803d-a377e2f059a5)
```
predicted value
```
![Screenshot 2024-10-10 150801](https://github.com/user-attachments/assets/11e28d0e-6033-4e74-b6b0-21f2490630d7)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
