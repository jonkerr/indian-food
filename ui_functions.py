from recommendation_models import filter_recipes
import streamlit as st
import numpy as np


# Function to reset session state
def reset_session_state():
    """
    Reset all session state variables, including clearing the uploaded image.
    """
    st.session_state.clear()  # Clear all session state
    st.session_state['uploaded_image'] = None  # Clear uploaded image
    st.session_state['file_uploader_key'] = str(np.random.randint(1, 1e9))  # Reset file uploader with a unique key
    st.session_state['predicted_dish_name'] = "Select an option"
    st.session_state['unformatted_dish_name'] = None
    st.session_state['prediction_done'] = False
    st.session_state['reset_done'] = True
    st.session_state['selected_recipe'] = "Select an option"
    st.session_state['selected_cuisine'] = "Select an option"
    st.session_state['selected_course'] = "Select an option"
    st.session_state['selected_diet_type'] = "Select an option"
    st.session_state['selected_prep_time'] = "Select an option"
    st.session_state['selected_allergies'] = []
    st.session_state['feedback_dict'] = {}
    st.session_state['recommended_recipes'] = None
    st.session_state['user_inputs'] = None
    st.session_state['reset_trigger'] = True  # Set a trigger for resetting widgets

    print("Session state after reset:", st.session_state)  # Debugging step

# Function to format dish names
def format_dish_name(dish_name):
    print('test1')
    # Remove special characters (replace underscores with spaces)
    predicted_dish_name_for_filter = dish_name.replace("_", " ")
    formatted_name = dish_name.replace("_", " ").title()
    return formatted_name, predicted_dish_name_for_filter

#Find dish name in recipes and filter them based on user preferences using filter_recipe() function
def filter_empty_option_and_df(df, dish_name, cuisine=None, course=None, diet=None, prep_time=None, allergen_type=None, debug=False): 

        cuisine = None if cuisine == "Select an option" else cuisine
        course = None if course == "Select an option" else course
        diet = None if diet == "Select an option" else diet
        prep_time = None if prep_time == "Select an option" else prep_time
        allergen_type = set(allergen_type) if allergen_type else None

        filtered_df = filter_recipes(df, dish_name, cuisine, course, diet, prep_time, allergen_type)

        return filtered_df

def format_instructions(instructions):
    # Replace all occurrences of "\xa0" with a regular space in the entire text
    instructions = instructions.replace("\xa0", " ")
    # Replace ".," with ". " to properly format sentences
    instructions = instructions.replace(".,", ". ")
    # Split instructions into sentences at ". "
    instructions_list = instructions.split(". ")
    # Ensure any lingering "\xa0" characters are removed in each sentence after the split
    instructions_list = [sentence.replace("\xa0", " ").strip() for sentence in instructions_list]
    # Re-check and clean the last sentence after joining, just in case
    instructions_list = [sentence.strip() for sentence in instructions_list]
    # Generate formatted instructions as a list of bullet points
    formatted_instructions = "\n".join(f"- {instruction}" for instruction in instructions_list if instruction)

    return formatted_instructions