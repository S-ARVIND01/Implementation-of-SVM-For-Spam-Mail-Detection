# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.

## Program:
Program to implement the SVM For Spam Mail Detection.

Developed by: ARVIND S

RegisterNumber:  212222240012
```python
#import packages
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("spam.csv",encoding="latin-1")
df.head()

#checking the data information and null presence
df.info()
df.isnull().sum()

#assigning x and y array
x=df["v1"].values
y=df["v2"].values

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#converting to numerical count in train and test set
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

#predicting y- i.e detecting spam
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

#checking the accuracy of the model
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
#### data.head():
![Screenshot 2023-11-01 104848](https://github.com/S-ARVIND01/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707337/53b303aa-30d8-41c5-8d85-b4fb1f3cdae4)

#### data.info():
![Screenshot 2023-11-01 104934](https://github.com/S-ARVIND01/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707337/a0ce7d5d-5a4d-4b39-9524-e390ad65357e)

#### data.isnull().sum():
![Screenshot 2023-11-01 104954](https://github.com/S-ARVIND01/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707337/137910d2-a60c-4871-80fc-903a7aa32c33)

#### Y_Prediction:
![Screenshot 2023-11-01 105049](https://github.com/S-ARVIND01/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707337/e97ff095-1447-4c17-8279-2842b457e8aa)

#### Accuracy Value:
![Screenshot 2023-11-01 105118](https://github.com/S-ARVIND01/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707337/6ea2608a-9756-48ba-8c49-fba1b19ea20a)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
