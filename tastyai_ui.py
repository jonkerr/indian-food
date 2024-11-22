import streamlit as st
from PIL import Image
import pandas as pd
import os
import numpy as np
import tempfile
from cv_predict import TastyFoodPredictor
from recommendation_models import get_recommendations_svd_tfidf  # Import functions

def main():
    # Load DataFrames from pickle files
    processed_df = pd.read_pickle("data/processed_recipes.pkl")

    # Load the pre-trained image-based model
    tasty_model = TastyFoodPredictor()

    # Define path for recipe images
    recipe_images_path = "data/indian_food_images/" 

    # Function to format dish names
    def format_dish_name(dish_name):
        print('test1')
        # Remove special characters (replace underscores with spaces)
        predicted_dish_name_for_filter = dish_name.replace("_", " ")
        formatted_name = dish_name.replace("_", " ").title()
        return formatted_name, predicted_dish_name_for_filter

    # Function to reset session state
    def reset_session_state():
        st.session_state['predicted_dish_name'] = st.session_state.get('predicted_dish_name', "Select an option")
        st.session_state['unformatted_dish_name'] = st.session_state.get('unformatted_dish_name', None)
        st.session_state['prediction_done'] = False
        st.session_state['reset_done'] = False
        st.session_state['selected_recipe'] = "Select an option"
        st.session_state['selected_cuisine'] = "Select an option"
        st.session_state['selected_course'] = "Select an option"
        st.session_state['selected_diet_type'] = "Select an option"
        st.session_state['selected_prep_time'] = "Select an option"
        st.session_state['selected_allergies'] = []

    # Initialize session state variables
    if 'predicted_dish_name' not in st.session_state:
        print('test2')
        reset_session_state()

    # Title and instructions
    st.title("TastyAI: Indian Recipe Recommendation System")
    st.write("Upload an image of a dish or select a dish name from the given list, select Cuisine, select course, select diet type, select preparation time, and select any allergy information for recommendations.")

    print('test3')

    # 1. Image Upload (Optional)
    uploaded_image = st.file_uploader("Upload an Image (optional)", type=["jpg", "jpeg", "png"])

    # Display uploaded image and provide an "Identify Dish Name" button
    if uploaded_image is not None:
        print('test4')
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Button to identify dish name using the pre-trained model
        if st.button("Identify Dish Name"):
            print('test5')
            # Save the uploaded image as a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file:
                temp_image_path = temp_image_file.name
                image.save(temp_image_path)

            try:
                # Predict the dish name using the path to the temporary image
                predicted_name = tasty_model.predict(temp_image_path)
                formatted_name, unformatted_name = format_dish_name(predicted_name)
                st.session_state['predicted_dish_name'] = formatted_name
                st.session_state['unformatted_dish_name'] = unformatted_name
                st.session_state['prediction_done'] = True
                st.success(f"Predicted Dish Name: {formatted_name}")
            finally:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)

        print('test6')

    # Add a label "OR" between the image uploader and the dropdown
    st.markdown("<h3 style='text-align: center;'>OR</h3>", unsafe_allow_html=True)

    # 2. Dish name Selection Dropdown
    dish_options = ["Select an option"] + [
        format_dish_name(name)[0] for name in processed_df["name"].unique()
    ]

    # Ensure dropdown uses the formatted predicted dish name
    if st.session_state['predicted_dish_name'] in dish_options:
        selected_dish_index = dish_options.index(st.session_state['predicted_dish_name'])
    else:
        selected_dish_index = 0

    # Dropdown logic
    dropdown_disabled = st.session_state['prediction_done'] and not st.session_state['reset_done']
    selected_recipe = st.selectbox(
        "Select a dish name",
        dish_options,
        index=selected_dish_index,
        disabled=dropdown_disabled,
        key="selected_recipe"
    )

    # Add a label for optional parameters
    st.markdown("<h4>Optional Parameters for Finding Recipes</h4>", unsafe_allow_html=True)

    # 3. Other Filters (Cuisine, Course, Diet Type, Preparation Time, Allergies)
    selected_cuisine = st.selectbox(
        "Select a Cuisine (optional)",
        ["Select an option"] + list(processed_df["cuisine"].unique()),
        key="selected_cuisine"
    )
    selected_course = st.selectbox(
        "Select a Course (optional)",
        ["Select an option"] + list(processed_df["course"].unique()),
        key="selected_course"
    )
    selected_diet_type = st.selectbox(
        "Select a Diet Type (optional)",
        ["Select an option"] + list(processed_df["diet"].unique()),
        key="selected_diet_type"
    )
    selected_prep_time = st.selectbox(
        "Select preparation time (optional)",
        ["Select an option"] + list(processed_df['categorized_prep_time'].unique()),
        key="selected_prep_time"
    )
    selected_allergies = st.multiselect(
        "Allergy Information",
        processed_df['allergen_type'].explode().unique(),
        key="selected_allergies"
    )

    def filter_recipes (df, dish_name, cuisine=None, course=None, diet=None, prep_time=None, allergen_type=None, debug=False): 
  
        print('test7')
        filtered_df = df[df['name'].str.contains(dish_name, case=False, na=False)]
        print ((f"Number of found dishes before filtering : {filtered_df.shape[0]}"))

        if cuisine:
            filtered_df = filtered_df[filtered_df['cuisine'] .str.lower() == cuisine.lower()]
        if course:
            filtered_df = filtered_df[filtered_df['course'].str.lower() == course.lower()]
        if diet:
            filtered_df = filtered_df[filtered_df['diet'] .str.lower() == diet.lower()]
        if prep_time:
            filtered_df = filtered_df[filtered_df['categorized_prep_time'] == prep_time]
        if allergen_type:
            # Exclude recipes with allergens in user preference
            allergen_set = set(allergen_type)
            filtered_df = filtered_df[~filtered_df['allergen_type'].apply(lambda x: bool(set(x) & allergen_set))]

        print('test8')

        if filtered_df.empty:
            print("No recipes found matching the criteria.")
            return None
        print(f"Number of dishes after filtering: {filtered_df.shape[0]}")
        return filtered_df


    # Button to get recommendations
    if st.button("Find Recipe"):
        if st.session_state['selected_recipe'] == "Select an option" and not st.session_state['prediction_done']:
            st.error("Please upload an image or select a dish name.")
        else:
            st.write("Fetching recommendations based on your input...")
            # Use predicted dish name if no selection is made
            dish_name_to_use = (
                st.session_state['unformatted_dish_name']
                if st.session_state['selected_recipe'] == "Select an option" else format_dish_name(st.session_state['selected_recipe'])[1]
            )

            print(dish_name_to_use)
            print(processed_df.head())

            # Filter the recipes using filter_recipes()
            filtered_df = filter_recipes(
                processed_df,
                dish_name_to_use,
                cuisine=st.session_state['selected_cuisine'],
                course=st.session_state['selected_course'],
                diet=st.session_state['selected_diet_type'],
                prep_time=st.session_state['selected_prep_time'],
                allergen_type=st.session_state['selected_allergies']
            )

            print(filtered_df.head())

            # filtered_df = processed_df.head()

            if filtered_df is None or filtered_df.empty:
                st.write("No recipes match your filters.")
            else:
                # Get recommendations using get_recommendations_svd_tfidf
                recommended_recipes = get_recommendations_svd_tfidf(filtered_df)

                if not recommended_recipes.empty:
                    st.write("Recommended Recipes:")
                    for _, row in recommended_recipes.iterrows():
                        formatted_name, _ = format_dish_name(row['name'])
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

        # Add "Search Another Recipe" button
        if st.button("Search Another Recipe"):
            reset_session_state()
            st.experimental_rerun()

if __name__ == '__main__':
    main()
