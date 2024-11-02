import streamlit as st
import pickle

# Function to load a model from a pickle file
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Check if the model is already in session state
if 'model' not in st.session_state:
    # Load default model from a pickle file
    default_model_path = '/mount/src/sed690-project2-running-group-704-710-712-714/RandomForest with out imbalance.pkl'  # Replace with your model file path
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
    st.write("Current model loaded:", st.session_state.model['Model name'])
else:
    st.write("No model loaded.")
