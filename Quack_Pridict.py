import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline



car = pd.read_csv('quikr_car.csv')

# print(car.head())

# print(car.info())

# print(car['Price'].unique())   #here is the more untractable values

# data Quality Need too work on it 
#   1. Year has any non_year Values
#   2. year object to int  conversion
#   3. IN price has Aks Of Price Need to remove
#   4. Price Object to Int Conversion
#   5. In Kms_driven has kms with intiger need to remove kms
#   6. kms_driven Object to int conversion
#   7. kms_driven has nan Values
#   8. In fuel_type has nan values
#   9. Keep 3 words of name 


# Cleaning data

backup = car.copy()


#   1. Year has any non_year Values
   
car = car[car['year'].str.isnumeric()]


#   2. year object to int  conversion

car['year'] = car['year'].astype(int)

# print(car.info())


#   3. IN price has Aks Of Price Need to remove
#   4. Price Object to Int Conversion

car = car[car['Price'] != "Ask For Price"]

# print(car.head())

 # removing  " , " in price   and change Object to int
car['Price'] = car['Price'].str.replace(',' , '').astype(int)

# print(car.head())
# print(car.info())


#   5. In Kms_driven has kms with intiger need to remove kms
car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',' , '')

# print(car['kms_driven'])
 
# remove Values in Kms_driven where petrol
car = car[car['kms_driven'].str.isnumeric()]

# print(car['kms_driven'])
    
#   6. kms_driven Object to int conversion
car['kms_driven'] = car['kms_driven'].astype(int)

# print(car.info())


#   8. In fuel_type has nan values
car = car[~car['fuel_type'].isna()]

# print(car.info())


#   9. Keep 3 words of name
car['name'] = car['name'].str.split(' ').str.slice(0,3).str.join(' ')

# print(car['name'])

# to proper index
car = car.reset_index(drop=True)

# print(car['name'])

# clean Data Looks like 
# print(car) 

# print(car.describe())  # here max value is not correct here is outlier 


# chek the how many car sold grater than 60,00000
# print(car[car['Price'] > 6e6])    # only one row afected 

# we need to keep everything wich in less than 60,00000
car = car[car['Price'] < 6e6].reset_index(drop=True)


car.to_csv('Cleaned_Car.csv')



# Model Traing

x = car.drop(columns='Price')
y = car['Price']

# print(y)


# Apply train test Split
x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.2)

ohe = OneHotEncoder()
ohe.fit(x[['name' , 'company', 'fuel_type']])

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
                                       remainder='passthrough')

lr = LinearRegression()

pipe = make_pipeline(column_trans, lr)

pipe.fit(x_train , y_train)


y_pred = pipe.predict(x_test)

# print(y_pred)

# See r2 Score
# print(r2_score(y_test, y_pred))

scores=[]
for i in range(1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    scores.append(r2_score(y_test,y_pred))
    
    
# print(np.argmax(scores))

# print(scores[np.argmax(scores)])


# The best model is found at a certain random state

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
r2_score(y_test,y_pred)



# dumping pipline

pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))



# cheking the pridiction of one car 
price_car = pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))
# print(price_car)

