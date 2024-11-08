import json
import streamlit as st
import pickle


# กำหนด URL หรือเส้นทางของภาพพื้นหลัง 
background_image_url = "https://wallpaperboat.com/wp-content/uploads/2020/12/14/63760/gears-28-920x518.jpg"
# กำหนดสีที่ต้องการ 
text_color = "#000000" 
# สีที่คุณต้องการ 
# ใส่ CSS สำหรับพื้นหลังและสีตัวอักษร 
st.markdown( f""" <style> .stApp {{ background-image: url('{background_image_url}'); background-size: cover; background-position: center; height: 100vh; }} h1, h2, h3, p, div {{ color: {text_color} !important; }}.block-container{{background-color: rgb(225 225 225 / 50%);box-shadow: 2px 5px 15px rgba(0, 0, 0, 0.3);}} </style> """, unsafe_allow_html=True )

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
    default_model_path = '/mount/src/sed690-project2-running-group-704-710-712-714/default3.pkl'  # Replace with your model file path
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
                    format="%d",  # Ensure integer format
                    step=1,  # Increment by 1 for integers
                    key=feature,
                    value=0  # Default value, can be adjusted as needed
                )
            else:  # For float values
                input_value = st.number_input(
                    label=f"Enter value for {feature}:",
                    format="%f",
                    key=feature,
                    value=0.0  # Default value for float
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
        
        # st.write("**Prediction Result:**", prediction[0])  # Display the prediction
        target_variable = st.session_state.model['target']
        if target_variable in label_encoders:
            # Decode the prediction using the label encoder
            prediction_label = label_encoders[target_variable].inverse_transform(prediction)[0]
        elif prediction[0] == 0:
            prediction_label = st.session_state.model['class0_label']
        elif prediction[0] == 1:
            prediction_label = st.session_state.model['class1_label']
        else:
            # If no label encoder, just use the raw prediction
            prediction_label = prediction[0]
        
        # Display the prediction result
        st.write("**Prediction Result:**", prediction_label) 
else:
    st.write("No model loaded.")
