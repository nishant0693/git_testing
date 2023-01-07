import streamlit as st
import numpy as np
import pandas as pd
import pickle
with open('nishant','rb') as file:
  pred=pickle.load(file)

st.title("Model Deployment in MAchine Learning")
a=st.number_input("Enter sepal length")
b=st.number_input("Enter sepal width")
c=st.number_input("Enter petal length")
d=st.number_input("Enter petal width")
result = pred.predict([[a, b, c, d]])
st.write(result)
