import streamlit as st             
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Dominant Color Extraction ")
st.subheader("Input Image")
img = st.file_uploader("Choose An Image") 

if img is not None:  
    st.subheader("Original Image")
    st.image(img) 

    print(type(img)) 

    img = plt.imread(img) 
    print(img.shape)

    n = img.shape[0]*img.shape[1]
    all_pixels= img.reshape((n, 3))

    k = st.number_input('Enter the no. of dominant colors',min_value=1)

    #create and fit model
    model  = KMeans(n_clusters = k) 
    model.fit(all_pixels)

    #creating centers
    centers = model.cluster_centers_.astype('uint8') 

    #creating new image
    new_img=np.zeros((n,3),dtype='uint8')

    for i in range(n): 
      group_idx = model.labels_[i]
      new_img[i] = centers[group_idx]

    new_img = new_img.reshape(*img.shape) 

    st.subheader("New Image")
    st.image(new_img) 