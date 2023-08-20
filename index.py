import streamlit as st

import numpy as np # linear algebra
import pandas as pd
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import statsmodels.api as sm
import pickle
from sklearn.linear_model import LinearRegression

# df = pd.read_csv("multiple_linear_regression_dataset.csv")

# df.info()

# print(df)
# print("")



# #Prepare Independent and Dependent Data 

# X = df[['age','experience']]
# y = df['income']

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=1)

# # Print the shape of splitted data
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# import LinearRegression from sklearn
# Model instantiation:(Creating LinearRegression Object)
# model = LinearRegression()

# # Fit the model using fit() function
# model.fit(X_train, y_train)

# #print Coefficient and Intercept
# # Print the intercept and coefficients
# print(model.intercept_)
# print(model.coef_)

# # Making predictions on the testing set
# y_pred = model.predict(X_test)

# #Compare Acutal and Predict Data
# d=pd.DataFrame({'Actual':y_test,"Predict":y_pred})
# print(d.head(10))

# # Save the model to a file
# with open('model_incomePredict.pkl', 'wb') as model_file:
#     pickle.dump(model, model_file)

st.title("Group 8")
st.header("Income Predictor")

# Load the saved model from a file
with open('model_incomePredict.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

st.write("In this app, you can predict income according to age and experience.")

age = st.number_input("age", step=1)
experience = st.number_input("experience", step=0.1, format="%.1f")
btn = st.button("Predict income")
if btn:
    if(age < 16):
        st.warning("Age shouldn't be less than 16. Please enter age again.")
    else:
        user_input = np.array([[age,experience]])
        predict_income = loaded_model.predict(user_input)
        st.write(f"Income prediction based your input:     $","%.4f" % predict_income[0])