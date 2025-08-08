import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('trained_model.pkl', 'rb'))

st.title('Sales Prediction App')
st.write("This app predicts the sales of a product based on various input features.")

st.sidebar.header('Input Product Features')

# All input fields, matching model feature order
item_identifier = st.sidebar.text_input("Item Identifier", "FDA15")
item_weight = st.sidebar.slider("Item Weight", 0.0, 25.0, 10.0)
item_fat_content = st.sidebar.selectbox("Item Fat Content", ["Low Fat", "Regular"])
item_visibility = st.sidebar.slider("Item Visibility", 0.0, 0.3, 0.05)
item_type = st.sidebar.selectbox("Item Type", [
    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables",
    "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
    "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
    "Breads", "Starchy Foods", "Others", "Seafood"
])
item_mrp = st.sidebar.slider("Item MRP", 0.0, 300.0, 150.0)
outlet_identifier = st.sidebar.text_input("Outlet Identifier", "OUT049")
outlet_est_year = st.sidebar.slider("Outlet Establishment Year", 1985, 2010, 2000)
outlet_size = st.sidebar.selectbox("Outlet Size", ["Small", "Medium", "High"])
outlet_location_type = st.sidebar.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
outlet_type = st.sidebar.selectbox("Outlet Type", [
    "Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"
])

# Simple encoders for categorical features (adjust these to match your training logic if needed!)
fat_map = {"Low Fat": 0, "Regular": 1}
size_map = {"Small": 0, "Medium": 1, "High": 2}
loc_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
outlet_type_map = {
    "Grocery Store": 0,
    "Supermarket Type1": 1,
    "Supermarket Type2": 2,
    "Supermarket Type3": 3
}
item_type_map = {v: i for i, v in enumerate([
    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
    "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast",
    "Health and Hygiene", "Hard Drinks", "Canned", "Breads", "Starchy Foods",
    "Others", "Seafood"
])}

# Prepare input in the expected order
input_data = pd.DataFrame([[
    item_identifier,                # String
    item_weight,                    # Float
    fat_map[item_fat_content],      # Encoded
    item_visibility,                # Float
    item_type_map[item_type],       # Encoded
    item_mrp,                       # Float
    outlet_identifier,              # String
    outlet_est_year,                # Int
    size_map[outlet_size],          # Encoded
    loc_map[outlet_location_type],  # Encoded
    outlet_type_map[outlet_type],   # Encoded
]], columns=[
    'Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
    'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
])

# Quick label encoding for string IDs (for demo - ideally use the same encoding as during training)
input_data['Item_Identifier'] = pd.factorize([item_identifier])[0]
input_data['Outlet_Identifier'] = pd.factorize([outlet_identifier])[0]

# Only keep numeric columns for the model
input_data_model = input_data.select_dtypes(include=["number"])

if st.button("Predict Sales"):
    prediction = model.predict(input_data_model)
    st.success(f"💰 Predicted Sales: ₹ {round(prediction[0], 2)}")
