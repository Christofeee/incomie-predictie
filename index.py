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
import streamlit as st



#title - used to add the title of an app
st.title("Group 8")

#header
st.header("Income Predictor")

df = pd.read_csv("multiple_linear_regression_dataset.csv")

df.info()

print(df)
print("")



#Prepare Independent and Dependent Data 

X = df[['age','experience']]
y = df['income']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=1)

# Print the shape of splitted data
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# import LinearRegression from sklearn
from sklearn.linear_model import LinearRegression

# Model instantiation:(Creating LinearRegression Object)
model = LinearRegression()

# Fit the model using fit() function
model.fit(X_train, y_train)

#print Coefficient and Intercept
# Print the intercept and coefficients
print(model.intercept_)
print(model.coef_)

# Making predictions on the testing set
y_pred = model.predict(X_test)

#Compare Acutal and Predict Data
d=pd.DataFrame({'Actual':y_test,"Predict":y_pred})
print(d.head(10))

# # Save the model to a file
# with open('model_incomePredict.pkl', 'wb') as model_file:
#     pickle.dump(model, model_file)

# Load the saved model from a file
with open('model_incomePredict.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

st.title("Loaded Model Details")
st.write("You've loaded a machine learning model!")

# Display model details (you can customize this part based on your model)
st.write("Model Type:", type(loaded_model))
st.write("Model Parameters:", loaded_model.get_params())

# Input feature values
feature1 = st.number_input("Feature 1:", min_value=0.0, max_value=10.0, value=5.0)
feature2 = st.number_input("Feature 2:", min_value=0.0, max_value=10.0, value=5.0)

# Make a prediction
prediction = loaded_model.predict([[feature1, feature2]])

st.write("Prediction:", prediction)












# #subheader
# st.subheader("bae jus said tht she hates me T.T ... hee she said it in cute way tho keke")

# #information 
# st.info("SO yeah THIS IS MY SHIT down there")

# #warning
# st.warning("U r going to Die in 3mins . . ....")

# #write
# st.write("write a name tht u wanna kill here. .. .. .x")

# n1 = st.number_input('Number a:')
# st.title("+")
# n2 = st.number_input('Number b:')
# st.title("=")

# result = n1+n2

# st.write('the combination of ',n1,' and ',n2,' is ',result)



