import streamlit as st
import pandas as pd
import pickle

# Load the trained Random Forest model
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Load the saved encoders
with open('ordinal_encoders.pkl', 'rb') as encoder_file:
    encoders = pickle.load(encoder_file)

# Function to preprocess input data before prediction
def preprocess_input(property_type, sector, bedRoom, bathroom, balcony, agePossession, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'property_type': [property_type],
        'sector': [sector],
        'bedRoom': [bedRoom],
        'bathroom': [bathroom],
        'balcony': [balcony],
        'agePossession': [agePossession],
        'built_up_area': [built_up_area],
        'servant room': [servant_room],
        'store room': [store_room],
        'furnishing_type': [furnishing_type],
        'luxury_category': [luxury_category],
        'floor_category': [floor_category]
    })

    # Encode categorical columns using loaded encoders
    for col, encoder in encoders.items():
        input_data[col] = encoder.transform(input_data[[col]])

    return input_data

# Function to predict price based on input data
def predict_price(input_data):
    return rf_model.predict(input_data)

# Create a Streamlit app
st.title('Gurgaon Real Estate Price Predictor')

# Input fields for user to enter property details
property_type = st.selectbox('Property Type', ['Flat', 'House'])
sector = st.selectbox('Sector', ['Sector 36', 'Sector 89', 'Sohna Road', 'Sector 92', 'Sector 102', 'Gwal Pahari', 'Sector 108', 'Sector 26', 'Sector 28', 'Sector 109', 'Sector 65', 'Sector 85', 'Sector 70A', 'Sector 30', 'Sector 107', 'Sector 3', 'Sector 2', 'Sector 62', 'Sector 49', 'Sector 81', 'Sector 66', 'Sector 86', 'Sector 48', 'Sector 51', 'Sector 37', 'Sector 111', 'Sector 67', 'Sector 113', 'Sector 13', 'Sector 61', 'Sector 69', 'Sector 67A', 'Sector 37D', 'Sector 82', 'Sector 53', 'Sector 74', 'Sector 52', 'Sector 12', 'Sector 95', 'Sector 41', 'Sector 83', 'Sector 104', 'Sector 88A', 'Sector 50', 'Sector 84', 'Sector 91', 'Sector 76', 'Sector 82A', 'Sector 78', 'Manesar', 'Sector 93', 'Sector 7', 'Sector 71', 'Sector 56', 'Sector 110', 'Sector 33', 'Sector 70', 'Sector 103', 'Sector 90', 'Sector 43', 'Sector 79', 'Sector 14', 'Sector 112', 'Sector 22', 'Sector 59', 'Sector 99', 'Sector 9', 'Sector 58', 'Sector 77', 'Sector 106', 'Sector 25', 'Sector 105', 'Dwarka Expressway', 'Sector 63', 'Sector 57', 'Sector 4', 'Sector 72', 'Sector 47', 'Sector 38', 'Sector 5', 'Sector 68', 'Sector 60', 'Sector 39', 'Sector 63A', 'Sector 11', 'Sector 24', 'Sector 46', 'Sector 17', 'Sector 15', 'Sector 10', 'Sector 55', 'Sector 6', 'Sector 21', 'Sector 80', 'Sector 31', 'Sector 73', 'Sector 54', 'Sector 45', 'Sector 1', 'Sector 88', 'Sector 40', 'Sector 23'])
bedRoom = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=2)
bathroom = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
balcony = st.number_input('Number of Balconies', min_value=0, max_value=5, value=1)
agePossession = st.selectbox('Age of Possession', ['New Property', 'Relatively New', 'Moderately Old', 'Old Property', 'Under Construction', 'Undefined'])
built_up_area = st.number_input('Built-up Area (in sqft)', min_value=100, max_value=10000, value=1000)
servant_room = st.checkbox('Servant Room')
store_room = st.checkbox('Store Room')
furnishing_type = st.selectbox('Furnishing Type', ['Non-Furnished', 'Semi-Furnished', 'Fully-Furnished'], format_func=lambda x: '0' if x == 'Non-Furnished' else '1' if x == 'Semi-Furnished' else '2')
luxury_category = st.selectbox('Luxury Category', ['Low', 'Medium', 'High'])
floor_category = st.selectbox('Floor Category', ['Low Floor', 'Mid Floor', 'High Floor'])

# When the user clicks the 'Predict Price' button
if st.button('Predict Price'):
    # Preprocess input data
    input_data = preprocess_input(property_type, sector, bedRoom, bathroom, balcony, agePossession, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category)
    
    # Predict the price
    predicted_price = predict_price(input_data)
    
    # Display the predicted price
    st.success(f'Predicted Price: ${predicted_price[0]:,.2f}')

