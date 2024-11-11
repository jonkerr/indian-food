import streamlit as st
from PIL import Image
import pandas as pd
import os
import tensorflow as tf
import numpy as np
from cv_predict import TastyFoodPredictor
import tempfile

# # Import get_recommendations from topic_modeling_NMF
# from topic_modeling_NMF import get_recommendations

# Load DataFrames from pickle files
cuisine_df = pd.read_pickle("data/nohindi_recipes.pkl")
recom_df = pd.read_pickle("data/processed_recipes.pkl")

# # Load the pre-trained model
tasty_model = TastyFoodPredictor()

# print(recom_df.columns)

# Define path for recipe images and model for indentifying dish name from image
recipe_images_path = "data/indian_food_images/" 
# model_path = "data/efficientnet_v2_20_84.64.keras"


# # Allergy options for checkboxes
allergy_options = recom_df['categorized_prep_time'].unique()
# print(allergy_options)

# Flatten allergen_type column to get unique allergens
allergen_options = set()
for allergens in recom_df['allergen_type']:
    if isinstance(allergens, list):  # Check if the value is a list
        allergen_options.update(allergens)  # Add each allergen in the list to the set
    elif isinstance(allergens, str):  # If there's a single string (not in list form)
        allergen_options.add(allergens)

# Convert the set to a sorted list for display
allergen_options = sorted(allergen_options)

# Title and instructions
st.title("TastyAI: Indian Recipe Recommendation System")
st.write("Upload an image of a dish or select dish name from the given list, select Cuisine, select course, select diet type, select preparation time, and select any allergy information for recommendations.")

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
            predicted_dish_name = tasty_model.predict(temp_image_path)
            
            # Display the identified dish name
            st.success(f"Identified Dish Name: {predicted_dish_name}")
        
        finally:
            # Clean up by deleting the temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

        # Optional: Automatically select the identified dish name in the dropdown
        selected_recipe = predicted_dish_name

# 2. Dish name Selection Dropdown
selected_recipe = st.selectbox("Select a dish name", cuisine_df["name"].unique())

# 3. Cuisine name Selection Dropdown
selected_cuisine = st.selectbox("Select a Cuisine (optional)", cuisine_df["cuisine"].unique())

# 4. Course name Selection Dropdown
selected_course = st.selectbox("Select a Course (optional)", cuisine_df["course"].unique())

# 5. Diet Type Selection Dropdown
selected_diet_type = st.selectbox("Select a Diet Type (optional)", cuisine_df["diet"].unique())

# 6. Preparation Time Selection Dropdown
selected_prep_time = st.selectbox("Select preparation time (optional)", allergy_options)

# 7. Allergy Information Checkboxes (Optional)
# 7. Allergy Information Checkboxes (Optional)
st.write("Select any allergy information (optional):")
selected_allergies = st.multiselect("Allergy Information", allergen_options)

# # Button to get recommendations
# if st.button("Find Recipe"):
#     # Check if either an image or a recipe is provided
#     if uploaded_image is None and not selected_recipe:
#         st.error("Please upload an image or select a recipe from the dropdown.")
#     else:
#         st.write("Fetching recommendations based on your input...")

#         # Apply filtering based on user preferences
#         filtered_df = filter_recipes(
#             recom_df,
#             cuisine=selected_cuisine,
#             course=selected_course,
#             diet=selected_diet_type,
#             prep_time=selected_prep_time,
#             allergen_type=selected_allergies
#         )

#         # Check if the filtered DataFrame is empty
#         if filtered_df.empty:
#             st.write("No recipes match your filters.")
#         else:
#             # Calculate topic matrix for the filtered DataFrame
#             topic_matrix = np.array(filtered_df["topic_vector"].tolist())

#             # Call get_recommendations with the selected dish name and filtered DataFrame
#             recommended_recipes = get_recommendations(
#                 dish_name=selected_recipe,
#                 df=filtered_df,
#                 topic_matrix=topic_matrix,
#                 num_recommendations=5
#             )

#             # Display the recommended recipes
#             if recommended_recipes is not None:
#                 st.write("Recommended Recipes:")
#                 for _, row in recommended_recipes.iterrows():
#                     st.subheader(row['name'])

#                     # Check for an image file corresponding to the recipe name
#                     image_path = os.path.join(recipe_images_path, f"{row['name']}.jpg")
#                     if os.path.exists(image_path):
#                         recipe_image = Image.open(image_path)
#                         st.image(recipe_image, caption=row['name'], use_column_width=True)

#                     # Show additional details if available
#                     st.write(f"Cuisine: {row['cuisine']}")
#                     st.write(f"Course: {row['course']}")
#                     st.write(f"Diet Type: {row['diet']}")
#                     st.write(f"Preparation Time: {row['prep_time']}")
#                     st.write(f"Ingredients: {row['cleaned_ingredients']}")
#                     st.write("Allergens:", row['allergens'])
#                     st.write("")

#             else:
#                 st.write("No recommendations available based on the selected options.")