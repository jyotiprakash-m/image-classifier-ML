import tensorflow as tf
from tensorflow import keras
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from skimage.transform import resize

classification = ['airplane', 'autombile', 'bird', 'cat',
                  'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_model():
    model = keras.models.load_model('./my_model.hdf5')
    return model


model = load_model()

st.write("""

    # Image classifier

""")
st.write('This program can classify the image between airplane, autombile, bird, cat, deer, dog, frog,  horse, ship and truck')

file = st.file_uploader('Please upload an image', type=['jpg', 'jpeg', 'png'])

if file is None:
    st.text("Please upload an Image File")
else:
    new_image = plt.imread(file)
    resize_image = resize(new_image, (32, 32, 3))
    predictions = model.predict(np.array([resize_image]))
    list_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x = predictions
    for i in range(10):
        for j in range(10):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    final_result = 'This is a image of ' + classification[list_index[0]]
    st.success(final_result)
    for i in range(5):
        st.write(classification[list_index[i]], ":: ", round(
            predictions[0][list_index[i]]*100, 2), " %")

    st.image(file, use_column_width=True)
