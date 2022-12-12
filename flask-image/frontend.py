import streamlit as st
import requests
from PIL import Image
from tensorflow.keras.preprocessing import image

input_image = st.file_uploader("Please upload an image", type=["jpg", "png"])
if input_image:

    # Display Input Image
    uploaded_image = Image.open(input_image)
    st.image(uploaded_image, caption='Your Uploaded Image')
    st.write('')

    # Resize
    uploaded_image = uploaded_image.resize((300, 200))
    
    # Convert PIL Image to NumPy Array
    uploaded_image = image.img_to_array(uploaded_image)

    # Convert NumPy Arrat to List
    uploaded_image = uploaded_image.tolist()
    
    if st.button('Predict'):
        URL = 'http://192.168.1.9:5000/predict'
        r = requests.post(url= URL, json={'user_image': uploaded_image})
        
        if r.status_code == 200:
            res = r.json()
            st.write('# The given image has been classified as : ', res['label_names'])
        else:
            st.write('Error with status code ', str(r.status_code))