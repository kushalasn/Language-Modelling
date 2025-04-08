import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import pickle

cars=pd.read_csv("C:/Users/kusha/OneDrive/Desktop/Task/Deep_Learning_Everyday/auto-mpg.csv")



cars['horsepower'] = cars['horsepower'].fillna(cars['horsepower'].mean())

X=cars.iloc[:,1:].values
print(X.shape)
y=cars.iloc[:,0].values


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

multilinear=LinearRegression()
multilinear.fit(X_train,y_train)
pickle.dump(multilinear,open('mlr.pkl','wb'))

#loading the saved model

model=pickle.load(open('mlr.pkl','rb'))

y_pred=model.predict(X_test)
y_pred

accuracy=r2_score(y_pred,y_test)
print(accuracy)
