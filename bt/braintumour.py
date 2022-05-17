"""
Analysis of Top Cryptocurrencies

"""
from random import random
import time
# from tkinter import N
# from turtle import color
# import _tkinter
# import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from urllib.request import urlopen
# from live import LivePrices
# rom analysis import ReturnTable, PlottingFunction

#---------------------------------------------------------------------------------------------------------------#
#                                                   PAGE SETTINGS                                               #
#---------------------------------------------------------------------------------------------------------------#

# Setting page configuration
st.set_page_config(page_title = "Brain Tumour Detection and Classification", page_icon = 
":hospital:", layout="wide")
st.markdown("##")

#---------------------------------------------------------------------------------------------------------------#
#                                                 SIDEBAR SETTINS                                               #
#---------------------------------------------------------------------------------------------------------------#

# Sidebar population
sBox = st.sidebar.selectbox("Menu", ["Home","Overview", "Analysis", "Detection","Recommendation", "Conclusion", "The Team"])
#st.sidebar.selectbox("Topics", ["Popularity", "Trends","Stability", "Growth", "Risk"])

#----------------------------------------- Sidebar options configuration ----------------------------------------

# Function to configure the Overview page specifications
def PageSpecifications(sBox):
    
    # ---------------------------------------------------- HOME PAGE---------------------------------------------------------------
    if sBox == "Home":
        # Setting the page header
        st.markdown("<h1 style='text-align: center; color: black;'> Brain Tumour Detection and Classification </h1>", unsafe_allow_html=True)
        # Adding spaces between the header and image
        st.write()
        st.write()
        #st.title("Cryptocurrency Performance Index")

        # Loading the homepage image
        col1, col2, col3 = st.columns([3.1,5,1])

        with col1:
            st.write('')

        with col2:
            imagemain = Image.open('Brain_scan.png')
            st.image(imagemain, caption="MRI Scan", width = 500)
            #st.markdown("##")

        with col3:
            st.write(' ')
        

        # Introduction
        st.write("")
        st.write("")
        st.markdown("<h3 style='text-align: center; color: black;'> Introduction </h3>", unsafe_allow_html=True)
        st.write("""Glioma, meningiomas and pituitary tumours are the most common types of brain tumours. This project seeks to 
        classify detected brain tumours from MRI scans inot one of the aforementioned categories.""")

           
    
    # -------------------------------------------------- OVERVIEW PAGE ---------------------------------------------------------------
    elif sBox == "Overview":
        # Setting the page header
        st.markdown("<h2 style='text-align: center; color: black;'> Overview </h2>", unsafe_allow_html=True)
        # Using columns to center image
        col1, col2, col3 = st.columns([3,6,1])

        with col1:
            # Column used to pad the image
            st.write("")

        with col2:
            # Loading page image
            imageOverview = Image.open(r"C:\\Users\\HP\\OneDrive\\Desktop\\Moringa\\brain tumour\\bt\\images\\overview.png")
            st.image(imageOverview, caption="Benign Vs Malignant", width= 400)
         

        with col3:
            # Column used to pad the image
            st.write("")
        
        # Creating space between the image and text body
        st.write("")
        st.write("")

        # Business overview
        # Paragraph one

        st.write("""Brain tumor is the growth of abnormal cells in the tissues of the brain.
         Brain tumors can be benign (non cancerous) or malignant (cancerous cells).""")


        # Paragraph two
        st.write( """
        The most common type of primary tumor is the glioma tumor. About 33\% of all brain tumors are gliomas.
         The tumor originates in the glial cells that surround and support neurons in the brain.
          Meningiomas are also common, and form 30\% of all brain tumors.
           The tumor arises from the meninges, therefore, technically, it is not a brain tumor but is included in this category as it 
           compresses the adjacent brain, nerves and blood vessels.
            According to the, a majority of the tumors are benign.
             The final example of a primary tumor is a pituitary tumor. They are abnormal growths that develop in the pituitary gland. The tumor can
              cause over or underproduction of hormones that regulate important functions of the body. Most pituitary tumors are benign and do not 
              spread to other parts of the body.""")

        #st.subheader("Cryptocurrency Overview")
    # ---------------------------------------------------- DETECTION PAGE------------------------------------------------------------

    elif sBox == "Detection":
        # setting header
        st.subheader("Detection")

        # Pre-trained model
        model = tf.keras.models.load_model('model/effnet.h5')
        # Loading image
        uploaded_file = st.file_uploader('Choose an image file', type = ['png', 'jpg', 'jpeg'])

        # Mapping dictionary
        map_dict = { 0: 'Glioma Tumour', 1: 'Meningioma Tumour', 2: 'No Tumour', 3: 'Pituitary Tumour'}
        if uploaded_file is not None:
            # Convert file to an opencv image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(opencv_image,(150, 150))
            # Displaying the image
            st.image(opencv_image, channels = 'RGB', width= 300, caption='Uploaded Image')
            # Button
            Genrate_pred = st.button("Generate Prediction") 
            if Genrate_pred:
                with st.spinner('Model working....'):
                    resized = tf.keras.applications.efficientnet.preprocess_input(resized)
                    # Reshaping the image
                    img_reshape = resized[np.newaxis,...]
                    prediction = model.predict(img_reshape).argmax()
                    time.sleep(1)
                    st.success('Classified')
                    if prediction == 2:
                        st.markdown("<h5 style='text-align: left; color: black;'> No tumour has been detected </h5>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h5 style='text-align: left; color: black;'> {} Detected </h5>".format(map_dict [prediction]), unsafe_allow_html=True)
   

    # ---------------------------------------------------- ANALYSIS PAGE---------------------------------------------------------------
    

    # Analysis page
    elif sBox == "Analysis":
        # Setting the page header
        st.markdown("<h2 style='text-align: center; color: black;'> Analysis </h2>", unsafe_allow_html=True)
        # Creating the analysis sections
        analysisRad = st.sidebar.selectbox("Topics", ["Tumour Types", "Tumour Frequency Plot"]) 

        
    # ---------------------------------------------------- RECOMMENDATION PAGE------------------------------------------------------------

    elif sBox == "Recommendation":
        # setting header
        st.subheader("Recommendation")

    # ------------------------------------------------------ CONCLUSION PAGE---------------------------------------------------------------

    elif sBox == "Conclusion":
        # setting header
        st.subheader("Conclusion")

    # --------------------------------------------------------TEAM MEMBERS ---------------------------------------------------------------
    elif sBox == "The Team":
        # Team Members and Roles
        st.markdown("<h4 style='text-align: left; color: black;'> Team Members </h4>", unsafe_allow_html=True)
    


# ---------------------------------------------------- SIDEBAR FUNTION CALL ---------------------------------------------------------------

PageSpecifications(sBox)

#------------------------------------------------------------------------------------------------------------------------------------------

    
    
