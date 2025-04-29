

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score 
## Program:
```
import pandas as pd
df=pd.read_csv("/content/Employee.csv")
print("data.head():")
df.head()

print("data.info()")
df.info()

print("data.isnull().sum()")
df.isnull().sum()

print("data value counts")
df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
df["salary"]=le.fit_transform(df["salary"])
df.head()

print("x.head():")
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

print("Data prediction")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt,filled=True,feature_names=x.columns,class_names=['salary' , 'left'])
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/d47eb413-977b-4fb9-9c63-7b817380f8ef)
![image-1](https://github.com/user-attachments/assets/72bb5996-7e85-4561-928c-5beab475c698)
![image-2](https://github.com/user-attachments/assets/399c547c-be14-4fd6-b204-91c99dc1ce73)
![image-3](https://github.com/user-attachments/assets/84ffac48-8ecf-448d-a986-e2b1bc27ce75)
![image-4](https://github.com/user-attachments/assets/21dcdb0d-31d9-42a3-82cb-f2147fda1ce0)
![image-5](https://github.com/user-attachments/assets/2a3a06d6-f332-499c-9661-f1e5f3e49c4a)
![image-6](https://github.com/user-attachments/assets/b6d3173a-e536-4c48-95cc-af994a05b3be)
![image-8](https://github.com/user-attachments/assets/8ef1b2c7-4873-46c9-88dc-5515bebf267b)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
