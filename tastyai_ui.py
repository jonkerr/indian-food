import streamlit as st
from PIL import Image
import pandas as pd
import os
import numpy as np
import pickle
from cv_predict import TastyFoodPredictor
import tempfile
from recommendation_models import find_filter_recipes, get_recommendations_nmf_tfidf  # Import functions

# Load DataFrames from pickle files
cuisine_df = pd.read_pickle("data/nohindi_recipes.pkl")
processed_df = pd.read_pickle("data/processed_recipes.pkl")

# # Load the NMF-TFIDF model and vectorizer
# with open("models/nmf_tfidf_model.pkl", "rb") as model_file:
#     nmf_model = pickle.load(model_file)
# with open("models/tfidf_vectorizer.pkl", "rb") as vectorizer_file:
#     tfidf_vectorizer = pickle.load(vectorizer_file)

# Load the pre-trained image-based model
tasty_model = TastyFoodPredictor()

# Define path for recipe images
recipe_images_path = "data/indian_food_images/" 

# Function to format dish names
def format_dish_name(dish_name):
    # Remove special characters (replace underscores with spaces)
    dish_name = dish_name.replace("_", " ")
    # Capitalize the first letter of each word
    formatted_name = dish_name.title()
    return formatted_name

# Title and instructions
st.title("TastyAI: Indian Recipe Recommendation System")
st.write("Upload an image of a dish or select a dish name from the given list, select Cuisine, select course, select diet type, select preparation time, and select any allergy information for recommendations.")

# Initialize session state for identified dish name and prediction status
if 'predicted_dish_name' not in st.session_state:
    st.session_state['predicted_dish_name'] = "Select an option"
if 'prediction_done' not in st.session_state:
    st.session_state['prediction_done'] = False

# 1. Image Upload (Optional)
uploaded_image = st.file_uploader("Upload an Image (optional)", type=["jpg", "jpeg", "png"])

# Display uploaded image and provide an "Identify Dish Name" button
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to identify dish name using the pre-trained model
    if st.button("Identify Dish Name"):
        # Save the uploaded image as a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file:
            temp_image_path = temp_image_file.name
            image.save(temp_image_path)

        try:
            # Predict the dish name using the path to the temporary image
            predicted_name = tasty_model.predict(temp_image_path)
            formatted_name = format_dish_name(predicted_name)
            st.session_state['predicted_dish_name'] = formatted_name
            st.session_state['prediction_done'] = True
            st.success(f"Predicted Dish Name: {formatted_name}")
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

# Add a label "OR" between the image uploader and the dropdown
st.markdown("<h3 style='text-align: center;'>OR</h3>", unsafe_allow_html=True)

# 2. Dish name Selection Dropdown
dish_options = ["Select an option"] + [format_dish_name(name) for name in cuisine_df["name"].unique()]

# Ensure dropdown uses the formatted predicted dish name
if st.session_state['predicted_dish_name'] in dish_options:
    selected_dish_index = dish_options.index(st.session_state['predicted_dish_name'])
else:
    selected_dish_index = 0

# Disable dropdown if a dish name is predicted from the uploaded image
disabled_dropdown = st.session_state['prediction_done']
selected_recipe = st.selectbox("Select a dish name", dish_options, index=selected_dish_index, disabled=disabled_dropdown)

# Add a label for optional parameters
st.markdown("<h4>Optional Parameters for Finding Recipes</h4>", unsafe_allow_html=True)

# 3. Other Filters (Cuisine, Course, Diet Type, Preparation Time, Allergies)
selected_cuisine = st.selectbox("Select a Cuisine (optional)", ["Select an option"] + list(cuisine_df["cuisine"].unique()))
selected_course = st.selectbox("Select a Course (optional)", ["Select an option"] + list(cuisine_df["course"].unique()))
selected_diet_type = st.selectbox("Select a Diet Type (optional)", ["Select an option"] + list(cuisine_df["diet"].unique()))
selected_prep_time = st.selectbox("Select preparation time (optional)", ["Select an option"] + list(processed_df['categorized_prep_time'].unique()))
selected_allergies = st.multiselect("Allergy Information", processed_df['allergen_type'].explode().unique())

# Button to get recommendations
if st.button("Find Recipe"):
    if selected_recipe == "Select an option" and not st.session_state['prediction_done']:
        st.error("Please upload an image or select a dish name.")
    else:
        st.write("Fetching recommendations based on your input...")
        # Use predicted dish name if no selection is made
        dish_name_to_use = (
            st.session_state['predicted_dish_name']
            if selected_recipe == "Select an option" else selected_recipe
        )

        # Filter the recipes using find_filter_recipes()
        filtered_df = find_filter_recipes(
            dish_name_to_use,
            processed_df,
            cuisine=selected_cuisine,
            course=selected_course,
            diet=selected_diet_type,
            prep_time=selected_prep_time,
            allergen_type=selected_allergies
        )

        if filtered_df.empty:
            st.write("No recipes match your filters.")
        else:
            # Get recommendations using get_recommendations_nmf_tfidf
            recommended_recipes = get_recommendations_nmf_tfidf(
                dish_name_to_use,
                filtered_df,
                tfidf_vectorizer,
                nmf_model,
                num_recommendations=5
            )

            if not recommended_recipes.empty:
                st.write("Recommended Recipes:")
                for _, row in recommended_recipes.iterrows():
                    formatted_name = format_dish_name(row['name'])
                    st.subheader(formatted_name)

                    image_path = os.path.join(recipe_images_path, f"{row['name']}.jpg")
                    if os.path.exists(image_path):
                        recipe_image = Image.open(image_path)
                        st.image(recipe_image, caption=formatted_name, use_column_width=True)

                    st.write(f"Cuisine: {row['cuisine']}")
                    st.write(f"Course: {row['course']}")
                    st.write(f"Diet Type: {row['diet']}")
                    st.write(f"Preparation Time: {row['prep_time']}")
                    st.write(f"Ingredients: {row['cleaned_ingredients']}")
                    st.write("Allergens:", row['allergens'])
                    st.write("")
            else:
                st.write("No recommendations available.")