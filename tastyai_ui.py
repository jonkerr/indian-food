import streamlit as st
import pandas as pd
import numpy as np
from feedback_reinf_learn_recommendation_model import FeedbackRecommendationModel
from ui_functions import (reset_session_state, format_dish_name, initialize_session_state, 
                          image_upload_and_prediction, dish_name_and_selection, 
                        display_optional_parameters, filter_empty_option_and_df, 
                        format_recipe_details_and_display, 
                        display_feedback_options_and_store_feedback_on_click, 
                        save_feedback_update_wt_refresh_screen)

# **** Overall flow of UI/TastyAI **** 
# This file create streamlit based UI for TastyAI. 
# For logic based interaction it logic is defined in various functions in UI_functions.py
# It loads cleaned dataframe from processed_recipes.pkl 
# displays all the fields and drop down options to the user using relevant data in processed_df
# calls TastyFoodPredictor for dish name identification when user uploads an image
# formats the dish name returned by TastyFoodPredictor in a presentable sentence case
# formats name in lower case to pass to the FeedbackRecommendationModel class to find recommended recipes 
# FeedbackRecommendationModel first calls NLP based compare_recommendations_model function which returns upto 5 dishes
# FeedbackRecommendationModel takes those recommended recipes and checks for relevant feedback for these dishes in user_feedback.json located in models folder
# if a matching recipe is found it aggregates the weight and combines with similarity score for final weighted score
# and then based on final_score returns re-ranked recipes to the user

# ******* Error Handling ******
# This UI also manages the error handling for cases where 1) recipe is not found 
# 2) user clicks find recipe button without selecting a dish name or uploading a picture
# 3) or uploads a picture but doesnt identify the dish name using "identify dish name" button
# 4) or an error is passed from compare_recommendations_model. 
# In all these cases appropriate message is displayed to the user
# 
# **** Limitations *****
# We had to use specific versions of python which was compatible with computer vision model
# Bacause of that some of the functionality from streamlit did not work. For eg:
# 1) tried using streamlit feedback_forms so that we can collect feedback for multiple/all displayed recipes at once
# but the click of the radio button was not recognized. Did not use that. 
# 2) tried using radio buttons without the forms, and there also radio button selection was not passed
# *** Used workarond: used on_click functionality within the radio button which was able to pass 
# which feedback option was passed by user and for which recipe. 
# but drawback is as soon as user selects an option all the displayed recipes refreshes. 
# But feedback is saved in ./models/user_feedback.json
# *** Also displayed a message before radio buttons what happens when the user clicks on that
# 
# 3) when "clear image" button is clicked reset_session is called to reset all the session variables and state
# but it is not working in one step. when user clicks on the button twice, it works.
# *** used workaround: displayed a message to the user on screen which is not ideal
# 
# 4) When "find another recipe" button is clicked at the end of all the recommended recipes, 
# it clears recommendations but does not clear user selections including uploaded image. 
# even though reset_session_state() is called to clear everything
# **** workaround used: displayed a message to the user that this may nor reset all the user selection on the screen

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

    # Initialize session state variables #added on dec 5
    if 'predicted_dish_name' not in st.session_state:
        reset_session_state()

    # Title and instructions
    st.title("TastyAI: Indian Recipe Recommendation System")
    st.write("Upload an image of a dish or select a dish name from the given list, select Cuisine, select course, select diet type, select preparation time, and select any allergy information for recommendations.")

    # Disclaimer in red font (to inform user about TastyFoodPredictor limitations)
    st.markdown(
        "<p style='font-size: small; color: blue;'><strong>Disclaimer:</strong> Dish identification may be incorrect because our model is trained only for 20 dishes.</p>",
        unsafe_allow_html=True)

    # 1 Display the upload button only if no image is uploaded or the clear button is clicked using image_upload_and_prediction() defined in ui_functions
    # this saves predicted dish name
    # perdicted dish name is formated in sentence case for the display, however it is saved in all lower case for filtering matching recipes
    image_upload_and_prediction()

    # Add a label "OR" between the image uploader and the dropdown
    st.markdown("<h3 style='text-align: center;'>OR</h3>", unsafe_allow_html=True)

    # 2. function to display dishes name drop down and saves user selected option in the session
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
            st.write("Fetching recommendations based on your input. It may take couple of minutes (upto 7 mins.)....")
            # Use predicted dish name if no selection is made
            dish_name_to_use = (
                st.session_state['unformatted_dish_name']
                if st.session_state['selected_recipe'] == "Select an option" else format_dish_name(st.session_state['selected_recipe'])[1]
            )

            # Filter the recipes using filter_empty_option_and_df() which calls filter_recipe() function
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
                    # st.experimental_rerun()
                    
            else:
                # Get recommendations using FeedbackRecommendationModel from feedback_reinf_learning_recommendation_model.py
                # FeedbackRecommendationModel calls compare_recommendations() functions to get upto top 5 recipes ranked by nmf/svd tfidf/count models
                # this function then uses user feedback into account to check if there is feedback matching user preferences, 
                # if yes, then apply average weights of all matching feedback in addition to similarity score calculated by compare_recommendations() 
                # then use combined score to rerank recommended recipe and then display final recommendations to the user
                if 'feedback_model' not in st.session_state:
                    st.session_state['feedback_model'] = FeedbackRecommendationModel(processed_df)
                
                # Ensure the 'combined_name_ingredients' column is present before passing to recommendation models
                if "combined_name_ingredients" not in filtered_df.columns:
                    # Create the combined_name_ingredients column if missing
                    if "name" in filtered_df.columns and "cleaned_ingredients" in filtered_df.columns:
                        filtered_df["combined_name_ingredients"] = (
                            filtered_df["name"] + " " + filtered_df["cleaned_ingredients"].apply(" ".join)
                        )
                    else:
                        # Raise an error if required columns are not available
                        raise KeyError("Required columns ('name', 'cleaned_ingredients') are missing to create 'combined_name_ingredients'")

                # Initialize the feedback model
                feedback_model = st.session_state['feedback_model']
                if 'stored_recommendations' not in st.session_state:
                    st.session_state['stored_recommendations'] = feedback_model.get_weighted_recommendations(filtered_df, user_inputs)
                
                if st.session_state['stored_recommendations'] is not None and not st.session_state['stored_recommendations'].empty:                    
                    st.write(f"Recommended Recipes ({len(st.session_state['stored_recommendations'])} found):")

                    # Initialize feedback storage in session state
                    if 'feedback_dict' not in st.session_state:
                        st.session_state['feedback_dict'] = {row.Index: "Select" for row in st.session_state['stored_recommendations'].itertuples()}

                    # Iterate over all recommended recipes and display details
                    for display_index, row in enumerate(st.session_state['stored_recommendations'].itertuples(), start=1):
                        st.write(f"Recipe {display_index} of {len(st.session_state['stored_recommendations'])}")  #sequential numbering

                        # Format the recipe details for display for each recipe
                        format_recipe_details_and_display(row)

                        # Instructions for the radio button selection (to manage the streamlit bug)     
                        st.markdown("<span style='color:red;'>App Limitation: You can only provide feedback for one recipe. Once you click on an option page will be refreshed</span>", unsafe_allow_html=True)

                        # display Feedback radio buttons after the recipe details are displayed and feedback selection stored
                        display_feedback_options_and_store_feedback_on_click(row, user_inputs, feedback_model)

                    # Instructions for the Submit button click (to manage the streamlit bug) 
                    st.markdown("<span style='color:red;'>Streamlit Limitation: Clicking on Find Another Recipe Button may not refresh previously uploaded image and option. Please use refresh option from the browser.</span>", unsafe_allow_html=True)
                    submitted = st.button("find another Recipe")
                        
                    # if user clicks on "find another Recipe" button
                    if submitted:
                        # process feedback and save it in a json file and refresh the screen
                        save_feedback_update_wt_refresh_screen(user_inputs, st.session_state['stored_recommendations'], feedback_model)

                        # after the feedback is saved and feedback file is saved with new weights, reset the session
                        reset_session_state()
                        st.session_state['feedback_dict'] = {}  # Clear feedback dictionary
                    
                else:
                    st.write("No recommendations available.")

if __name__ == '__main__':
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore'  # suppress warnings for cleaner output
    main()