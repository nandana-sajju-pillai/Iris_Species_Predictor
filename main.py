import streamlit as st
import pandas as pd
from os import path
import pickle

st.title("Flower Species Predictor")

petal_length = st.number_input("Petal Length (in millimeters)", min_value=1.0,max_value=6.9,
                               placeholder="Please enter a valid number between 1.0 and 6.9 ",value=None)
petal_width = st.number_input("Petal Width (in millimeters)",min_value=0.1,max_value=2.5,
                              placeholder="Please enter a valid number between 0.1 and 2.5 ",value=None)
sepal_length = st.number_input("Sepal Length (in millimeters)",min_value=4.3,max_value=7.9,
                               placeholder="Please enter a valid number between 4.3 and 7.9",value=None)
sepal_width = st.number_input("Sepal Width (in millimeters)",min_value=2.0,max_value=4.4,
                              placeholder="Please enter a valid number between 2.0 and 4.4",value=None)
# prepare the dataframe for prediction
df_user_input = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],
                          columns=['sepal_length','sepal_width', 'petal_length','petal_width'])

#using the .pkl file, creating an ML model named 'iris_predictor'
model_path = path.join("model","rf_model.pkl")
with open(model_path, 'rb') as file:
    iris_predictor=pickle.load(file)

dict_species={0:'setosa', 1:'versicolor', 2:'virginica'}

if st.button("Predict species"):
    if((petal_length==None) or (petal_width==None)
            or (sepal_length==None) or (sepal_width==None)):
        st.write("Please enter all values") # will be executed when any of the values is not entered properly
    else:
        # prediction can be done here. we are expecting a dataframe
        predicted_species = iris_predictor.predict(df_user_input)
        # predicted_species[0] will give us the value in the dataframe
        # we use that value to find the corresponding species from the dictionary 'dict_species'
        st.write("The Species is ", dict_species[predicted_species[0]])
