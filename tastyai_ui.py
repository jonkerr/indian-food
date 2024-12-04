import streamlit as st
from PIL import Image
import pandas as pd
import os
import numpy as np
import tempfile
from cv_predict import TastyFoodPredictor
from feedback_reinf_learn_recommendation_model import FeedbackRecommendationModel
from save_user_feedback import save_feedback
from ui_functions import reset_session_state, format_dish_name, filter_empty_option_and_df, format_instructions


def main():
    # Load DataFrames from pickle files
    processed_df = pd.read_pickle("data/processed_recipes.pkl")

    # Initialize session state for feedback
    if 'feedback' not in st.session_state:
        st.session_state['feedback'] = {}

    # Load the pre-trained image-based model
    tasty_model = TastyFoodPredictor()

    # Define path for recipe images
    recipe_images_path = "data/indian_food_images/" 

    # Initialize session state variables
    if 'predicted_dish_name' not in st.session_state:
        reset_session_state()

    # Title and instructions
    st.title("TastyAI: Indian Recipe Recommendation System")
    st.write("Upload an image of a dish or select a dish name from the given list, select Cuisine, select course, select diet type, select preparation time, and select any allergy information for recommendations.")

    # 1. Image Upload (Optional)
    if 'uploaded_image' not in st.session_state:
        st.session_state['uploaded_image'] = None  # Initialize session state for the uploaded image

    # Display the file uploader
    uploaded_image = st.file_uploader(
        "Upload an Image (optional)",
        type=["jpg", "jpeg"],
        key=st.session_state.get('file_uploader_key', 'default_uploader_key')
    )

    # Display uploaded image if present in session state
    if st.session_state.get('uploaded_image') is not None:
        image = Image.open(st.session_state['uploaded_image'])
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save uploaded image to session state
    if uploaded_image is not None:
        st.session_state['uploaded_image'] = uploaded_image

        # Button to identify dish name using the pre-trained model
        if st.button("Identify Dish Name"):
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

    # Add a label "OR" between the image uploader and the dropdown
    st.markdown("<h3 style='text-align: center;'>OR</h3>", unsafe_allow_html=True)

    # 2. Dish name Selection Dropdown
    dish_options = ["Select an option"] + [
        format_dish_name(name)[0] for name in processed_df["name"].unique()
    ]

    # Ensure dropdown uses the formatted predicted dish name
    if st.session_state.get('reset_trigger', False):
        st.session_state['selected_recipe'] = "Select an option"  # Reset state
        st.session_state['reset_trigger'] = False  # Clear the reset trigger

    # Dropdown is disabled only if a prediction is made and not reset
    dropdown_disabled = st.session_state['prediction_done'] and not st.session_state['reset_done']
    selected_recipe = st.selectbox(
        "Select a dish name",
        dish_options,
        index=dish_options.index(st.session_state.get('selected_recipe', "Select an option")),
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

    # Button to get recommendations
    if st.button("Find Recipe"):
        # Gather all user inputs
        user_inputs = {
            "dish_name": (
                st.session_state['unformatted_dish_name']
                if st.session_state['selected_recipe'] == "Select an option"
                else format_dish_name(st.session_state['selected_recipe'])[1]
            ),
            "selected_cuisine": st.session_state['selected_cuisine'],
            "selected_course": st.session_state['selected_course'],
            "selected_diet_type": st.session_state['selected_diet_type'],
            "selected_prep_time": st.session_state['selected_prep_time'],
            "selected_allergies": st.session_state['selected_allergies']
        }
        
        # Check if a dish name is provided
        if st.session_state['selected_recipe'] == "Select an option" and not st.session_state['prediction_done']:
            st.error("Please upload an image or select a dish name.")
        else:
            st.write("Fetching recommendations based on your input. It may take 3-7 minutes....")
            # Use predicted dish name if no selection is made
            dish_name_to_use = (
                st.session_state['unformatted_dish_name']
                if st.session_state['selected_recipe'] == "Select an option" else format_dish_name(st.session_state['selected_recipe'])[1]
            )

            # Filter the recipes using filter_empty_option_and_df()
            filtered_df = filter_empty_option_and_df(
                processed_df,
                dish_name_to_use,
                cuisine=st.session_state['selected_cuisine'],
                course=st.session_state['selected_course'],
                diet=st.session_state['selected_diet_type'],
                prep_time=st.session_state['selected_prep_time'],
                allergen_type=st.session_state['selected_allergies']
            )

            # print(filtered_df.head())

            if filtered_df is None or filtered_df.empty:
                st.write("No recipes match your filters.")
            else:
                # Get recommendations using FeedbackRecommendationModel
                if 'feedback_model' not in st.session_state:
                    st.session_state['feedback_model'] = FeedbackRecommendationModel(processed_df)
                
                # Initialize the feedback model
                feedback_model = st.session_state['feedback_model']

                recommended_recipes = feedback_model.get_weighted_recommendations(filtered_df, user_inputs)

                if recommended_recipes is not None and not recommended_recipes.empty:
                    st.write(f"Recommended Recipes ({len(recommended_recipes)} found):")

                    # Initialize feedback storage in session state if not already set
                    if 'feedback_dict' not in st.session_state:
                        st.session_state['feedback_dict'] = {row.Index: "Select" for row in recommended_recipes.itertuples()}

                    # Create a form to group feedback and submission together
                    with st.form("feedback_form"):
                        # Iterate over all recommended recipes and display details
                        for display_index, row in enumerate(recommended_recipes.itertuples(), start=1):
                            st.write(f"Recipe {display_index} of {len(recommended_recipes)}")  # Correct sequential numbering
                            print(f"Displaying recipe {display_index}: {row.name}")  # print

                            # Format the recipe name for display
                            formatted_name, _ = format_dish_name(row.name)
                            st.markdown(f"### **{formatted_name}**")

                            # Display recipe details
                            st.markdown(f"**<span style='color:blue;'>Cuisine:</span>** {row.cuisine}", unsafe_allow_html=True)
                            st.markdown(f"**<span style='color:blue;'>Course:</span>** {row.course}", unsafe_allow_html=True)
                            st.markdown(f"**<span style='color:blue;'>Diet Type:</span>** {row.diet}", unsafe_allow_html=True)
                            st.markdown(f"**<span style='color:blue;'>Preparation Time:</span>** {row.prep_time} minutes", unsafe_allow_html=True)

                            # Format and display the ingredients
                            ingredients = row.cleaned_ingredients
                            formatted_ingredients = "\n".join(f"- {ingredient}" for ingredient in ingredients)
                            st.markdown("**<span style='color:blue;'>Ingredients:</span>**", unsafe_allow_html=True)
                            st.markdown(formatted_ingredients)

                            # Format and display allergens as a comma-separated list
                            allergens = ", ".join(row.allergens)
                            st.markdown(f"**<span style='color:blue;'>Allergens:</span>** {allergens}", unsafe_allow_html=True)

                            # Instructions
                            st.markdown("**<span style='color:blue;'>Instructions:</span>**", unsafe_allow_html=True)
                            if row.instructions:
                                formatted_instructions = format_instructions(row.instructions)
                                # Display formatted instructions
                                st.markdown(formatted_instructions)
                            else:
                                st.write("Instructions not available.")

                            # Feedback radio buttons
                            feedback_key = f"feedback_{row.Index}"
                            selected_feedback = st.radio(
                                f"Was this recipe helpful?",
                                options=["Select", "Helpful", "Not Related"],
                                index=0,
                                key=feedback_key
                            )

                            if selected_feedback != "Select":
                                st.session_state['feedback_dict'][row.Index] = selected_feedback

                            st.write("---")  # Separator between recipes

                        # Submit all feedback button
                        submitted = st.form_submit_button("Submit feedback and/or find another Recipe")

                        if submitted:
                            if any(feedback != "Select" for feedback in st.session_state['feedback_dict'].values()):
                                feedback_data = {
                                    "user_inputs": user_inputs,
                                    "feedback": st.session_state['feedback_dict'],
                                    "recommendations": recommended_recipes.to_dict(orient="records")
                                }

                                # File path for the feedback file
                                feedback_file_path = "models/user_feedback.json"

                                # Ensure the file and directory exist
                                os.makedirs(os.path.dirname(feedback_file_path), exist_ok=True)

                                try:
                                    # Load existing feedback if the file exists
                                    with open(feedback_file_path, "r") as f:
                                        existing_feedback = json.load(f)
                                except FileNotFoundError:
                                    # If the file doesn't exist, initialize with an empty list
                                    existing_feedback = []

                                # Append new feedback data
                                existing_feedback.append(feedback_data)

                                # Save updated feedback back to the file
                                with open(feedback_file_path, "w") as f:
                                    json.dump(existing_feedback, f, indent=4)

                                # Update the model with the feedback
                                feedback_model.update_weights(st.session_state['feedback_dict'])
                                
                                print("Session state before reset:", st.session_state)
                                # Reset everything, including the uploaded image
                                reset_session_state()
                                st.experimental_rerun()  # Ensure the UI refreshes completely
                    
                else:
                    st.write("No recommendations available.")

if __name__ == '__main__':
    # Necessary to avoid multiprocessing issues
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore'  # suppress warnings for cleaner output
    main()