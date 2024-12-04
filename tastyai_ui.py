import streamlit as st
import pandas as pd
import numpy as np
from recommendation_models import filter_recipes
from feedback_reinf_learn_recommendation_model import FeedbackRecommendationModel
from save_user_feedback import save_feedback
from ui_functions import reset_session_state, format_dish_name, initialize_session_state, image_upload_and_prediction, dish_name_and_selection, \
    display_optional_parameters, filter_empty_option_and_df, format_recipe_details_and_display, display_feedback_radio_button_and_store_feedback, \
        save_feedback_update_wt_refresh_screen

def main():
    # Load DataFrames from pickle files
    processed_df = pd.read_pickle("data/processed_recipes.pkl")

    # Initialize session state for feedback
    if 'feedback' not in st.session_state:
        st.session_state['feedback'] = {}

    # Define path for recipe images
    recipe_images_path = "data/indian_food_images/" 

    # Initialize session state variables
    initialize_session_state()

    # Title and instructions
    st.title("TastyAI: Indian Recipe Recommendation System")
    st.write("Upload an image of a dish or select a dish name from the given list, select Cuisine, select course, select diet type, select preparation time, and select any allergy information for recommendations.")

    # Disclaimer in red font
    st.markdown(
        "<p style='font-size: small; color: red;'><strong>Disclaimer:</strong> Dish identification may be incorrect because our model is trained only for 20 dishes.</p>",
        unsafe_allow_html=True)

    # 1 Display the upload button only if no image is uploaded or the clear button is clicked using image_upload_and_prediction() defined in ui_functions
    # this saves predicted dish name
    # perdicted dish name is formated in sentence case for the display, however it is saved in all lower case for filtering matching recipes
    image_upload_and_prediction()

    # Add a label "OR" between the image uploader and the dropdown
    st.markdown("<h3 style='text-align: center;'>OR</h3>", unsafe_allow_html=True)

    # 2. Dish name Selection Dropdown, using dish_name_and_selection() function
    # function to display dishes name drop down and saves user selected option in the session
    dish_name_and_selection()

    # Add a label for optional parameters
    st.markdown("<h4>Optional Parameters for Finding Recipes</h4>", unsafe_allow_html=True)

    # 3. Other optional Filters (Cuisine, Course, Diet Type, Preparation Time, Allergies) 
    display_optional_parameters()

    # Button to find recipes based on user selected criteria
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

            # Filter the recipes using filter_recipes() drom recommendations_models.py
            filtered_df = filter_empty_option_and_df(
                processed_df,
                dish_name_to_use,
                cuisine=st.session_state['selected_cuisine'],
                course=st.session_state['selected_course'],
                diet=st.session_state['selected_diet_type'],
                prep_time=st.session_state['selected_prep_time'],
                allergen_type=st.session_state['selected_allergies']
            )

            if filtered_df is None or filtered_df.empty:
                st.write("No recipes match your filters.")
                if st.button("Find Another Recipe"):
                    reset_session_state()
                    
            else:
                # Get recommendations using FeedbackRecommendationModel from feedback_reinf_learning_recommendation_model.py
                # FeedbackRecommendationModel calls compare_recommendations() functions to get upto top 5 recipes ranked by nmf/svd tfidf/count models
                # this function then uses user feedback into account to check if there is feedback matching user preferences, 
                # if yes, then apply average weights of all matching feedback in addition to similarity score calculated by compare_recommendations() 
                # then use combined score to display final recipes to the user
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

                            # Format the recipe details for display for each recipe
                            format_recipe_details_and_display(row)

                            # display Feedback radio buttons after the recipe details are displayed and feedback selection stored
                            display_feedback_radio_button_and_store_feedback(row)

                            st.write("---")  # Separator between recipes

                        # Submit all feedback button
                        submitted = st.form_submit_button("Submit feedback and/or find another Recipe")

                        # if user clicks on "Submit feedback and/or find another Recipe" button
                        # process feedback and save it in a json file and 
                        # update the weights of the feedback
                        # refresh the screen
                        if submitted:
                            if any(feedback != "Select" for feedback in st.session_state['feedback_dict'].values()):
                                save_feedback_update_wt_refresh_screen()
                    
                else:
                    st.write("No recommendations available.")

if __name__ == '__main__':
    # Necessary to avoid multiprocessing issues
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore'  # suppress warnings for cleaner output
    main()