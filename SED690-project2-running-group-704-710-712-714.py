import json
import streamlit as st
import pickle

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
            number_format = "%d" if data_type['Data Type'] == 'int64' else "%f"
            input_value = st.number_input(
                label=f"Enter value for {feature}:",
                format=number_format,
                key=feature
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

        # Prepare input for prediction (ensure correct order and shape)
        input_data = [encoded_input[feature] for feature in st.session_state.model['features']]

        # Use the model to predict
        model = st.session_state.model['model']
        prediction = model.predict([input_data])  # Wrap input_data in a list for a single prediction
        
        # Display the feature values and the prediction result
        st.write("**Collected Input Values:**")
        for feature, value in input_values.items():
            st.write(f"{feature}: {value}")
        
        st.write("**Prediction Result:**", prediction[0])  # Display the prediction
else:
    st.write("No model loaded.")
