import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt # Import matplotlib
import pickle
import urllib.request


# Define the plot_learning_curves function (if not imported)
def plot_learning_curves(history):
  if hasattr (history, "history"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
  else:
    acc=history['accuracy']
    val_acc=history['val_accuracy']
    loss=history['loss']
    val_loss=history['val_loss']
    
  epochs_range = range(len(acc))

  fig, axes = plt.subplots(1, 2, figsize=(8, 8)) # Use subplots to get figure and axes

  axes[0].plot(epochs_range, acc, label='Training Accuracy')
  axes[0].plot(epochs_range, val_acc, label='Validation Accuracy')
  axes[0].legend(loc='lower right')
  axes[0].set_title('Training and Validation Accuracy')

  axes[1].plot(epochs_range, loss, label='Training Loss')
  axes[1].plot(epochs_range, val_loss, label='Validation Loss')
  axes[1].legend(loc='upper right')
  axes[1].set_title('Training and Validation Loss')

  return fig # Return the figure object

MODEL_URL="https://github.com/Jenny133573/garbage-classifier-app/releases/download/MobileNetV2/fine_tuned_Trash_classifier.keras"
MODEL_PATH="fine_tuned_Trash_classifier.keras"

# Load the trained model
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... (first run only)")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)  # Downloads directly to MODEL_PATH
    st.success("Model downloaded!")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# load the training history
@st.cache_resource
def load_history(file_path):
  script_dir=os.path.dirname(os.path.abspath(__file__))
  history_path=os.path.join(script_dir, file_path)
  try:
    with open(history_path, 'rb') as f:
      history=pickle.load(f)
      return history
  except FileNotFoundError:
    st.error("Please ensure the file is in the Github repo.")
    return None
  except Exception as e:
    st.error("Error loading training_history:", e)
    return None

history=load_history("initial_training_hitory.pkl")
# Define the class names based on the model's output (0 for not_recyclable, 1 for recyclable)
class_names = ['not_recyclable', 'recyclable']

st.title('Garbage Classification App')
st.markdown("Classify an object as trash or recyclable.")

# Assuming 'history' object is available from a previous training run
# If 'history' is not available, you would need to train the model within or before the Streamlit app
# For demonstration, we'll assume 'history' is available.
# In a real deployment, you might save the history object or regenerate the plot separately.


tab1, tab2, tab3=st.tabs([":bar_chart: Data", "Upload an image", "Take a photo"])


with tab1:
  st.header("Data")
  st.write("The dataset used for training was from the Garbage Classification (12 classes) Dataset from Kaggle")
  st.link_button("Visit Dataset Website", "https://www.kaggle.com/datasets/mostafaabla/garbage-classification")
  st.subheader("Visualization for training process before fine-tuning the model.")
  # Call the plotting function and display the plot
  # Assuming 'history' object is available globally or passed somehow
  # Replace 'history' with your actual history object from training
  try:
      fig = plot_learning_curves(history)
      st.pyplot(fig)
  except NameError:
      st.warning("Training history not available. Please run the training cell first.")
  st.subheader("Visulization for training process after fine-tuning the model.")


with tab2:
  st.header("Upload an image")
  uploaded_file = st.file_uploader("Choose an image...", type="jpg")

  if uploaded_file is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image.', use_container_width=True)

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class_index = int(np.round(prediction[0][0]))
    predicted_class = class_names[predicted_class_index]
    prediction_probability = prediction[0][0] # Get the probability

    st.write(f"Prediction: This item is likely **{predicted_class}**")
    st.write(f"Probability: **{prediction_probability:.2f}**") # Display the probability formatted to 2 decimal places

with tab3:
  st.header("Take a photo")
  camera_photo = st.camera_input("Take a photo")

  if camera_photo is not None:
    # Read the image from the camera input
    img = Image.open(camera_photo)
    img = img.resize((224, 224)) # Resize the image

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class_index = int(np.round(prediction[0][0]))
    predicted_class = class_names[predicted_class_index]
    prediction_probability = prediction[0][0] # Get the probability

    st.write(f"Prediction: This item is likely **{predicted_class}**")
    st.write(f"Probability: **{prediction_probability:.2f}**") # Display the probability formatted to 2 decimal places
