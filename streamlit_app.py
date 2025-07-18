def type_num(num):
  return np.float64(num)

def Hours_Studied_treatment(num):
  return type_num(num / 35)

def Attendance_treatment(num):
  return type_num(num / 100)

def low_medium_high(stringN):
  return type_num(-1) if stringN == 'Low' else type_num(0) if stringN == 'Medium' else type_num(1)

def yes_no(num):
  return type_num(-1) if num == 'No' else type_num(1)

def Sleep_Hours_treatment(num):
  return type_num(num / 10)

def Previous_Scores_treatment(num):
  return type_num(num / 100)

def Exam_Score(num):
  return type_num(num / 100)

def Tutoring_Sessions_treatment(num):
  return type_num(num / 4)

def School_Type_treatment(stringN):
  return type_num(1) if stringN == 'Private' else type_num(-1)

def Peer_Influence_treatment(stringN):
  return type_num(1) if stringN == 'Positive' else type_num(0) if stringN == 'Neutral' else type_num(-1)

def Physical_Activity_treatment(num):
  return type_num(num / 4)

def Parental_Education_Level_treatment(stringN):
  return type_num(-1) if stringN == 'High School' else type_num(0) if stringN == 'College' else type_num(1)

def Distance_from_Home_treatment(stringN):
  return type_num(1) if stringN == 'Near' else type_num(0) if stringN == 'Moderate' else type_num(-1)

def Gender_treatment(stringN):
  return type_num(1) if stringN == 'Female' else type_num(-1)

def switch_block(factor, num,q):
  match factor:
    case 'Hours_Studied':
      
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = Hours_Studied_treatment(feature_val)
      return r
     
    case 'Attendance':
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = Attendance_treatment(feature_val)
      return r
     
    case 'Parental_Involvement':
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = low_medium_high(feature_val)
      return r
     
    case 'Access_to_Resources':
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = low_medium_high(feature_val)
      return r
     
    case 'Extracurricular_Activities':
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = yes_no(feature_val)
      return r
     
    case 'Sleep_Hours':
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = Sleep_Hours_treatment(feature_val)
      return r
     
    case 'Previous_Scores':
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = Previous_Scores_treatment(feature_val)
      return r
     
    case 'Motivation_Level':
      feature_val = st.number_input(q, value=0.0, key=factor)

      r = low_medium_high(feature_val)
      return r
     
    case 'Internet_Access':
      feature_val = st.number_input(q, value=0.0, key=factor)

      r = yes_no(feature_val)
      return r
     
    case 'Tutoring_Sessions':
      feature_val = st.number_input(q, value=0.0, key=factor)

      r = Tutoring_Sessions_treatment(feature_val)
      return r
     
    case 'Family_Income':
      feature_val = st.number_input(q, value=0.0, key=factor)

      r = low_medium_high(feature_val)
      return r
     
    case 'Teacher_Quality':
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = low_medium_high(feature_val)
      return r
     
    case 'School_Type':
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = School_Type_treatment(feature_val)
      return r
     
    case 'Peer_Influence':
      
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = Peer_Influence_treatment(feature_val)
      return r
     
    case 'Physical_Activity':
      
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = Physical_Activity_treatment(feature_val)
      return r
     
    case 'Learning_Disabilities':
      
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = yes_no(feature_val)
      return r
     
    case 'Parental_Education_Level':
      
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = Parental_Education_Level_treatment(feature_val)
      return r
     
    case 'Distance_from_Home':
      
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = Distance_from_Home_treatment(feature_val)
      return r
     
    case 'Gender':
      
      feature_val = st.number_input(q, value=0.0, key=factor)
      r = Gender_treatment(feature_val)
      return r
     
    case _:
      return 0






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
    st.set_page_config(layout="centered", page_title="gradeS Predictor", page_icon=":bar_chart:")

    st.title("gradeS Predictor")
    st.markdown("""
   Predict your future grades, with one of the most accurate AI model for this task (90% accuracy).
    """)

    # --- Load Model Section ---
    st.header("1. Model Status")
    model_filename = "my_model.keras"
    model = None
    try:
        with st.spinner(f"Loading {model_filename}..."):
            model = load_keras_model(model_filename)
        st.success(f"Model '{model_filename}' loaded successfully!")
  
    except Exception as e:
        st.error(f"Failed to load the model. Please ensure '{model_filename}' exists and is valid. Error: {e}")
        st.stop() # Stop the app if model loading fails

    # --- Data Input Section ---
    st.header("2. Enter Data for Prediction")
    st.markdown(f"""
    I'll just ask you important questions that will determine the precision of my predictions. So, please : give me always the exact information, in order to give you a relevant result.
    """)

    # Create 10 input fields for features
    feature_values = []
    # Using a loop to create input fields for 10 features
    
    question_text = [
     
"Hours_Studied:	How many hours do you spend studying per week.",
"Attendance: What's your percentage of classes attended: like 92, or 99, or even 100 for the 🤓📕",
"Parental_Involvement:	Do your parents care about your studies? 👪 (Low, Medium, High).",
"Access_to_Resources:	Are you always able to afford educational resources 😬? (Low, Medium, High).",
"Extracurricular_Activities:   Do you do anything apart going to school like doing sport, music, painting, dancing...? (Yes, No).",
"Physical_Activity:	Sport is important for your body health as well for your mental health (I assure you!) ; So how many hours of physical activity do you do per week?",
"Sleep_Hours:	How many hours do you sleep per night 💤.",
"Previous_Scores:   What's the average of your previous exam scores? 0-100 if it's /20, just multiply it by 5.",
"Motivation_Level:	Are you like: I will do whatever I can to have the best school grade ('High'); I wanna get a good grade ('Medium'); Oh, again a school exam! ('Low'); (Low, Medium, High).",
"Internet_Access:	Probably obvious but decisive: do you have any internet access (Yes, No).",
"Tutoring_Sessions:	How many tutoring sessions did you attend per month ?",
"Family_Income:	Spicy question : What's your family income level (Low, Medium, High).",
"Teacher_Quality: Answer it frankly without being spiteful: How is the quality of your teachers (Low, Medium, High).",
"School_Type:  In which type of school are you ? (Public, Private).",
"Peer_Influence:  And, how your friends (peers) influence your academic performance ? Take your time, and think about it. (Positive, Neutral, Negative).",

"Learning_Disabilities:	Have you unfortunately any learning disabilities ? (Yes, No).",
"Parental_Education_Level: In which education level have ended your parents ? (High School, College, Postgraduate).",
"Distance_from_Home:  How far do you live from school ? (Near, Moderate, Far).",
"Gender: Are you a male (includes man, fresh man, boy, baby boy), or a female (includes woman, fresh woman, girl, baby girl) ? Sorry other genders are not accepted here, if it's your case you will have to say the truth ! (Male, Female)."

    ]


    factors = [
    "Hours_Studied",

"Attendance",

"Parental_Involvement",

"Access_to_Resources",

"Extracurricular_Activities",

"Sleep_Hours",

"Previous_Scores",

"Motivation_Level",

"Internet_Access",

"Tutoring_Sessions",

"Family_Income",

"Teacher_Quality",

"School_Type",

"Peer_Influence",

"Physical_Activity",

"Learning_Disabilities",

"Parental_Education_Level",

"Distance_from_Home",

"Gender"

 ]



    for i in range(19):
        # You can adjust default values as needed
        
        input1 = switch_block(factors[i], 0,question_text[i])
        feature_values.append(input1)

    # Convert to numpy array for prediction
    input_array = np.array(feature_values).reshape(1, 19).astype(np.float32)
    print(input_array.shape)

    # --- Prediction Button ---
    if st.button("Predict your future grades 🤗!"):
        if model is not None:
            try:
                # Make prediction
                prediction = model.predict(input_array[0])

                st.subheader("3. Prediction Result 🥁(drumroll please)")
               # st.write(f"Input Features: {input_array[0].tolist()}")
               # st.write(f"Raw Prediction: `{prediction[0].tolist()}`") # Show all outputs if multi-output

                # Optional: Add interpretation based on the output layer's activation
                # This assumes a single output neuron for binary classification (sigmoid)
                # or multiple for multi-class (softmax) or regression.
                
                index_result = np.argmax(prediction, axis=1)
                grade = index_result / 5
                if grade > 18:
                    st.success(f"Predicted Grade: **{grade[0]:.2f}** With these assets, you will succeed !")
                elif grade > 12:
                    st.success(f"Predicted Grade: **{grade[0]:.2f}** With these assets, you should succeed at school!")
                elif grade > 8:
                    st.warning(f"Predicted Grade: **{grade[0]:.2f}** Outch, you should change your strategy, now if possible 😧!")
                else:
                    st.error(f"Predicted Grade: **{grade[0]:.2f}** Aïe, outch 🤕, I hope that I am wrong at my predictions ! Try to change your study strategy or report an error (say our real exam scores)")

                 

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Please ensure your input data matches the model's expected input shape and type.")
        else:
            st.warning("Model not loaded. Please ensure 'my-model.keras' is in the same directory.")

if __name__ == "__main__":
    main()
