import json
import streamlit as st
import pickle
import numpy as np
import random

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
    default_model_path = '/mount/src/sed690-project2-running-group-704-710-712-714/default2.pkl'  # Replace with your model file path
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

    st.write("**Current model loaded:**", f"{st.session_state.model['Model name']}")
    st.write("**Evaluation**")
    st.write("**Accuracy:**", f"{st.session_state.model['Accuracy']:.3f}")
    st.write("**Precision:**", f"{st.session_state.model['Precision']:.3f}")
    st.write("**Recall:**", f"{st.session_state.model['Recall']:.3f}")
    st.write("**F1-Score:**", f"{st.session_state.model['F1-Score']:.3f}")

    st.write("**Features**")
    label_encoders = st.session_state.model['label_encoders']
    data_types = st.session_state.model['data_types']

    for feature in st.session_state.model['features']:
        data_type = data_types[feature]

        if feature in label_encoders:
            original_labels = label_encoders[feature].classes_
            selected_label = st.selectbox(f"Choose value for {feature}:", original_labels, key=feature)
            input_values[feature] = selected_label  # Store the input value in a dictionary
        elif len(data_type['Unique Values']) > 0:
            selected_label = st.selectbox(f"Choose value for {feature}:", data_type['Unique Values'], key=feature)
            input_values[feature] = selected_label  # Store the input value in a dictionary
        elif data_type['Data Type'] in ['int64', 'float64']:
            # Create a number input for this column
            if data_type['Data Type'] == 'int64':
                input_value = st.number_input(
                    label=f"Enter value for {feature}:",
                    format="%d",
                    step=1,
                    key=feature,
                    value=0
                )
            else:  # For float values
                input_value = st.number_input(
                    label=f"Enter value for {feature}:",
                    format="%f",
                    key=feature,
                    value=0.0
                )
            input_values[feature] = input_value  # Store the input value in a dictionary
        else:
            input_value = st.text_input(f"Enter value for {feature}:", key=feature)
            input_values[feature] = input_value  # Store the input value in a dictionary

    if st.button("Submit"):
        # Encode input values using label encoders
        encoded_input = {}
        for feature, value in input_values.items():
            if feature in label_encoders:
                # Use the label encoder to transform the selected label
                encoded_input[feature] = label_encoders[feature].transform([value])[0]
            else:
                # Directly use the input value for numerical types
                encoded_input[feature] = value

        # Prepare input for prediction
        input_data = [encoded_input[feature] for feature in st.session_state.model['features']]

        # Random sampling until prediction equals 1
        found_prediction = False
        random_feature_values = {}

        while not found_prediction:
            # Generate random values based on data types
            for feature in st.session_state.model['features']:
                if feature in label_encoders:
                    # For categorical features, randomly select from original labels
                    random_value = random.choice(label_encoders[feature].classes_)
                elif data_types[feature]['Data Type'] == 'int64':
                    # Random integer value (adjust the range as necessary)
                    random_value = random.randint(0, 100)  # Example range; modify as needed
                elif data_types[feature]['Data Type'] == 'float64':
                    # Random float value (adjust the range as necessary)
                    random_value = random.uniform(0.0, 100.0)  # Example range; modify as needed
                else:
                    # For text or other types, use a placeholder or random string
                    random_value = "random_string"  # Modify this as needed

                random_feature_values[feature] = random_value

            # Encode the random values for prediction
            random_encoded_input = {}
            for feature, value in random_feature_values.items():
                if feature in label_encoders:
                    random_encoded_input[feature] = label_encoders[feature].transform([value])[0]
                else:
                    random_encoded_input[feature] = value

            # Prepare input for prediction
            random_input_data = [random_encoded_input[feature] for feature in st.session_state.model['features']]
            prediction = st.session_state.model['model'].predict([random_input_data])  # Wrap in a list

            if prediction[0] == 1:
                found_prediction = True  # Stop if prediction equals 1

        # Display the feature values that produced a prediction of 1
        st.write("**Random Feature Values that resulted in Prediction of 1:**")
        for feature, value in random_feature_values.items():
            st.write(f"{feature}: {value}")

        st.write("**Prediction Result:**", prediction[0])  # Display the prediction
else:
    st.write("No model loaded.")
