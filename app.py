#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from predict_page import show_predict_page

# page = st.sidebar.selectbox("Predict", ("Predict"))
st.write('how many images you want to test')

#n=st.number_input("How could you elaborate NLP?")
#n=st.number_input(label='cholesterol level in mg/dL',step=1,format="%i")
# if page == "Predict":
#for i in range(n):
show_predict_page()
    


# In[ ]:




