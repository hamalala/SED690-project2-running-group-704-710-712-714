import json
import streamlit as st
import pickle
import numpy as np

# URL of the background image and color settings
background_image_url = "https://wallpaperboat.com/wp-content/uploads/2020/12/14/63760/gears-28-920x518.jpg"
text_color = "#000000"

# CSS for background and text color
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url('{background_image_url}');
            background-size: cover;
            background-position: center;
            height: 100vh;
        }}
        h1, h2, h3, p, div {{
            color: {text_color} !important;
        }}
        .block-container{{
            background-color: rgb(225 225 225 / 50%);
            box-shadow: 2px 5px 15px rgba(0, 0, 0, 0.3);
        }}
    </style>
    """, 
    unsafe_allow_html=True
)

# Function to load a model from a pickle file
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

st.title("Automate Imbalance Prediction")
st.write("Group: 704-710-712-714")
st.write("\n\n")

# Check if the model is already in session state
if 'model' not in st.session_state:
    # Load default model from a pickle file
    default_model_path = '/mount/src/sed690-project2-running-group-704-710-712-714/default3.pkl'
    st.session_state.model = load_model(default_model_path)
    st.session_state.model_loaded = True
    st.success("Loaded default model.")

# File uploader for new model
uploaded_file = st.file_uploader("Upload a new model (pickle file)", type=['pkl'])
if uploaded_file is not None:
    # Load the uploaded model
    st.session_state.model = pickle.load(uploaded_file)
    st.session_state.model_loaded = True
    st.success("New model uploaded and loaded successfully.")

# Display the model status
if st.session_state.get('model_loaded', False):
    input_values = {}
    label_encoders = st.session_state.model['label_encoders']
    data_types = st.session_state.model['data_types']
    
    # Button for randomizing input to produce class 1
    if st.button("Randomize to Get Class 1"):
        found_class1 = False
        attempt_limit = 100  # limit to avoid infinite loop
        attempt_count = 0
        
        while not found_class1 and attempt_count < attempt_limit:
            input_values = {}
            for feature in st.session_state.model['features']:
                data_type = data_types[feature]
                
                # Generate random values for each feature based on data type
                if feature in label_encoders:
                    original_labels = label_encoders[feature].classes_
                    input_values[feature] = np.random.choice(original_labels)
                elif len(data_type['Unique Values']) > 0:
                    input_values[feature] = np.random.choice(data_type['Unique Values'])
                elif data_type['Data Type'] in ['int64', 'float64']:
                    input_values[feature] = np.random.randint(0, 100) if data_type['Data Type'] == 'int64' else np.random.uniform(0, 1)
                else:
                    input_values[feature] = "Random Text"  # Default for text input
            
            # Encode input values using label encoders
            encoded_input = {}
            for feature, value in input_values.items():
                if feature in label_encoders:
                    encoded_input[feature] = label_encoders[feature].transform([value])[0]
                else:
                    encoded_input[feature] = value
            
            # Prepare input for prediction
            input_data = [encoded_input[feature] for feature in st.session_state.model['features']]
            prediction = st.session_state.model['model'].predict([input_data])
            
            # Check if prediction is class 1
            if prediction[0] == 1:
                found_class1 = True
            attempt_count += 1
        
        # Display the feature values if class 1 was found
        if found_class1:
            st.write("**Randomized Input Values for Class 1 Prediction:**")
            for feature, value in input_values.items():
                st.write(f"{feature}: {value}")
            st.write("**Prediction Result:** Class 1")
        else:
            st.warning("Unable to generate Class 1 within the attempt limit.")
