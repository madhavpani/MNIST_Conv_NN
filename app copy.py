# Importing Libraries

import streamlit as st
import numpy as np
import keras

# Loading the model
model = keras.models.load_model('model.keras')
(_, __), (X_test, y_test) = keras.datasets.mnist.load_data()
# Create the Page

# creating two cols
col1, col2 = st.columns([1, 3], gap='small', border=False, vertical_alignment='center')

# Insert an Image Under col1
with col1:
    st.image("Images/logo.png", width=100)

# Insert a Title Under Col2

with col2:
    st.write('## 🔢 **:blue[MNIST Data Classification]**')

# Creating an Expander

with st.expander(':blue[**ABOUT THE PROJECT**]'):

    # Writing About the Project
    st.write('🔢 **:blue[MNIST Data Classification]** is a **:green[DEEP LEARNING PROJECT]** that Classifies the **:green[Hand Written Digits]**.')
    st.write('**:blue[Convolutional Neural Network]** is used to build the model.')

# Show the Model

# create a header for the model
st.warning(f'**MODEL**', icon=':material/model_training:')

# create a container for input and output
with st.container():

    col1, col2 = st.columns(2, border=True,vertical_alignment='top')

    # use col1 for input
    with col1:
        st.info('**INPUT**', icon=':material/input:')

        index_col, img_col = st.columns([1.5,1], border=False, vertical_alignment='center')
        with index_col:
            index = st.text_input('**:blue[IMAGE INDEX]**', placeholder='0 - 9999')
        with img_col:
            if index:
                index = np.int64(index)
                st.image(X_test[index], width=100)
            else:
                index = 0
                st.image(X_test[index], width=100)

            input_image = np.expand_dims(X_test[index], axis=0)

    # Use col2 for output
    with col2:
        st.success('**OUTPUT**', icon=':material/output:')
        prediction = model.predict(input_image)
        L = list(max(prediction))
        st.write(f'# **:blue[THE DIGIT IS {L.index(max(L))}]**')

# container for like, madee, video, repository, connection, hf space
with st.container(border=False):
    col1, col2, col3, col4, col5, col6 = st.columns([.85,1,.95,.95,1.15,1.15], border=False, vertical_alignment='top')

    with col1:
        st.button('**LIKE**', icon=':material/favorite:')

    with col2:
        st.button('**MADEE**', icon=':material/flight:', disabled=True)

    with col3:
        st.link_button('**VIDEO**', icon=':material/slideshow:', url='https://youtu.be/QUgf2SZJys4')

    with col4:
        st.link_button('**REPO**', icon=':material/code_blocks:', url='https://github.com/madhavpani/MNIST_data_Classification')

    with col5:
        st.link_button('**CONNECT**', icon=':material/connect_without_contact:', url='https://www.linkedin.com/in/madhavpani')

    with col6:
        st.link_button('**HF SPACE**', icon=':material/sentiment_satisfied:', url='https://huggingface.co/spaces/madhav-pani/MNIST_data_Classification/tree/main')
