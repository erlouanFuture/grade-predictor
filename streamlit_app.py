# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import os

# Function to load the Keras model with caching
# @st.cache_resource is crucial for performance: it loads the model only once
# when the app starts, not every time a user interacts with it.
@st.cache_resource
def load_keras_model(model_path="my_model.keras"):
    """
    Loads a Keras model from the specified path.
    Uses st.cache_resource to load the model only once.
    """
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop() # Stop the app if the model isn't found
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        st.info("Please ensure the model is a valid Keras file and all necessary custom objects (if any) are handled.")
        st.stop() # Stop the app if model loading fails

def main():
    st.set_page_config(layout="centered", page_title="Keras Model Predictor (10 Features)")

    st.title("Keras Model Predictor")
    st.markdown("""
    This app uses your `my_model.keras` model to make predictions based on 10 numerical input features.
    """)

    # --- Load Model Section ---
    st.header("1. Model Status")
    model_filename = "my_model.keras"
    model = None
    try:
        with st.spinner(f"Loading {model_filename}..."):
            model = load_keras_model(model_filename)
        st.success(f"Model '{model_filename}' loaded successfully!")
        st.subheader("Model Summary:")
        # Capture model summary to display in Streamlit
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        st.text("\n".join(summary_list))

        # Explicitly set expected input features based on the user's request
        expected_num_features = 10
        st.info(f"The model expects input data with **{expected_num_features} features**.")

    except Exception as e:
        st.error(f"Failed to load the model. Please ensure '{model_filename}' exists and is valid. Error: {e}")
        st.stop() # Stop the app if model loading fails

    # --- Data Input Section ---
    st.header("2. Enter Data for Prediction")
    st.markdown(f"""
    Please enter {expected_num_features} numerical values for prediction.
    """)

    # Create 10 input fields for features
    feature_values = []
    # Using a loop to create input fields for 10 features
    for i in range(expected_num_features):
        # You can adjust default values as needed
        default_val = float(i + 1) / 10.0 # Example: 0.1, 0.2, ..., 1.0
        feature_val = st.number_input(f"Feature {i+1}", value=default_val, key=f"feature_input_{i}")
        feature_values.append(feature_val)

    # Convert to numpy array for prediction
    input_array = np.array(feature_values).reshape(1, -1).astype(np.float32)

    # --- Prediction Button ---
    if st.button("Predict Outcome"):
        if model is not None:
            try:
                # Make prediction
                prediction = model.predict(input_array)
                
                st.subheader("3. Prediction Result")
                st.write(f"Input Features: {input_array[0].tolist()}")
                st.write(f"Raw Prediction: `{prediction[0].tolist()}`") # Show all outputs if multi-output

                # Optional: Add interpretation based on the output layer's activation
                # This assumes a single output neuron for binary classification (sigmoid)
                # or multiple for multi-class (softmax) or regression.
                if prediction.shape[-1] == 1 and model.layers[-1].activation == tf.keras.activations.sigmoid:
                    st.write(f"Probability of Class 1: `{prediction[0][0]:.4f}`")
                    if prediction[0][0] > 0.5:
                        st.success("Predicted Class: **1**")
                    else:
                        st.info("Predicted Class: **0**")
                elif prediction.shape[-1] > 1 and model.layers[-1].activation == tf.keras.activations.softmax:
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    st.write(f"Predicted Class (Index): `{predicted_class}`")
                    st.write(f"Probabilities per class: `{prediction[0].tolist()}`")
                else:
                    st.info("No specific interpretation applied (e.g., Regression or custom activation).")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Please ensure your input data matches the model's expected input shape and type.")
        else:
            st.warning("Model not loaded. Please ensure 'my-model.keras' is in the same directory.")

if __name__ == "__main__":
    main()
