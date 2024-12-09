from recommendation_models import filter_recipes
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import os
import json
from cv_predict import TastyFoodPredictor

processed_df = pd.read_pickle("data/processed_recipes.pkl")

# Function to reset session state
def reset_session_state():
    """
    Reset all session state variables, including clearing the uploaded image.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]
        
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
    st.session_state['clear_image_triggered'] = True

def initialize_session_state():
    if 'predicted_dish_name' not in st.session_state:
        reset_session_state()
    if 'uploaded_image' not in st.session_state:
        st.session_state['uploaded_image'] = None
    if 'clear_image_triggered' not in st.session_state:
        st.session_state['clear_image_triggered'] = True
    if 'prediction_done' not in st.session_state:
        st.session_state['prediction_done'] = False
    if 'selected_recipe' not in st.session_state:
        st.session_state['selected_recipe'] = "Select an option"

# Function to format dish names
def format_dish_name(dish_name):
    # Remove special characters (replace underscores with spaces)
    predicted_dish_name_for_filter = dish_name.replace("_", " ")
    formatted_name = dish_name.replace("_", " ").title()
    return formatted_name, predicted_dish_name_for_filter

# function to display upload image option, clear image and/or predict dish name. this also saves predicted dish name
# perdicted dish name is formated in sentence case for the display, however it is saved in all lower case for filtering matching recipes
def image_upload_and_prediction():
    # Load the pre-trained image-based model
    tasty_model = TastyFoodPredictor()

    if st.session_state.get('uploaded_image') is None:
        # st.session_state['clear_image_triggered'] = False
        uploaded_image = st.file_uploader("Upload an Image (optional)", type=["jpg", "jpeg"], 
                                          accept_multiple_files=False, key="image_uploader",
                                          help="Limit 200MB per file • JPG, JPEG only")
        if uploaded_image is not None:
            # Get the uploaded file's extension
            file_extension = uploaded_image.name.split(".")[-1].lower()
            
            # check if the file extension is anything other than .jpg or .jpeg then display an error
            if file_extension not in ["jpg", "jpeg"]:
                st.error("Invalid file format. Please upload JPG or JPEG format only.")
                st.stop()  # Stop further execution
            else:
                st.session_state['uploaded_image'] = uploaded_image
                # st.session_state['clear_image_triggered'] = False  # Reset clear trigger

    # Display uploaded image if present in session state
    if st.session_state.get('uploaded_image') is not None:
        st.markdown("### Uploaded Image:")
        image = Image.open(st.session_state['uploaded_image'])
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Clear Image Button
        if st.button("Clear Image"):
            print("Before reset:", st.session_state)  # Debug print
            reset_session_state()
            print("After reset:", st.session_state)  # Debug print
            # st.experimental_rerun()

        # Instructions for the button (to manage the bug)
        st.markdown(
            "<p style='font-size: small; color: gray;'><em>*Please click this button twice with a 2-second interval to clear the image.*</em></p>",
            unsafe_allow_html=True
        )

        # preidict dish name
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
                st.session_state["selected_recipe"] = "Select an option"  # Reset dropdown value
                st.success(f"Predicted Dish Name: {formatted_name}")
            finally:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)

# function to display dishes name drop down and saves user selected option in the session
def dish_name_and_selection():
    dish_options = ["Select an option"] + [
        format_dish_name(name)[0] for name in processed_df["name"].unique()
    ]

    # Dropdown is disabled only if a prediction is made
    dropdown_disabled = st.session_state.get("prediction_done", False)

    # # Dropdown is disabled only if a prediction is made and not reset
    selected_recipe = st.selectbox(
        "Select a dish name",
        dish_options,
        index=dish_options.index(st.session_state.get('selected_recipe', "Select an option")),
        disabled=dropdown_disabled,
        key="selected_recipe"
    )

# this function displays all the optional parameters using the options available in our limited recipe dataset (processed_df)
# all options are single selection, except allergy type which is multi select
def display_optional_parameters():
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
        "Allergy Information (optional and multi-select field)",
        processed_df['allergen_type'].explode().unique(),
        key="selected_allergies"
    )

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
    instructions_list = [sentence.replace("\\xa0", " ").strip() for sentence in instructions_list]
    instructions_list = [sentence.replace("\xa0", " ").strip() for sentence in instructions_list]
    # Re-check and clean the last sentence after joining, just in case
    instructions_list = [sentence.strip() for sentence in instructions_list]
    # Generate formatted instructions as a list of bullet points
    formatted_instructions = "\n".join(f"- {instruction}" for instruction in instructions_list if instruction)

    return formatted_instructions

# function to Format the recipe name for display for each recipe
def format_recipe_details_and_display(row):
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
    # allergens = ", ".join(row.allergens)
    if isinstance(row.allergens, str):
        allergens = row.allergens  # If it's a single string, use it as is
    else:
        allergens = ", ".join(row.allergens)  # Otherwise, join the list of allergens
    st.markdown(f"**<span style='color:blue;'>Allergens:</span>** {allergens}", unsafe_allow_html=True)

    # Instructions
    st.markdown("**<span style='color:blue;'>Instructions:</span>**", unsafe_allow_html=True)
    if row.instructions:
        formatted_instructions = format_instructions(row.instructions)
        # Display formatted instructions
        st.markdown(formatted_instructions)
    else:
        st.write("Instructions not available.")

# on-click of the radio button for feedback, extract the user selected dish name
# which option user selected, and then save it in feedback dictionary
def on_click_extract_feedback_to_dict(row, user_inputs, feedback_model):
    # Identify the feedback key for the current row
    fbIndex = f"fb_{row.Index}"
    # Retrieve the selected feedback value from session state
    selected_feedback_value = st.session_state.get(fbIndex, None)

    # Ensure feedback_dict exists in session state
    if 'feedback_dict' not in st.session_state:
        # Initialize feedback_dict with "Select" for all recommendations
        st.session_state['feedback_dict'] = {
            r.Index: "Select" for r in st.session_state['stored_recommendations'].itertuples()
        }

    # Update the feedback_dict for the given row
    if selected_feedback_value is not None and selected_feedback_value != "Select":
        st.session_state['feedback_dict'][row.Index] = selected_feedback_value
    else:
        print("No valid feedback was selected.")

    save_feedback_update_wt_refresh_screen(row, user_inputs, feedback_model)

# # display Feedback options after the recipe details are displayed
def display_feedback_options_and_store_feedback_on_click(row, user_inputs, feedback_model):
    # print(row)

    feedback_key = f"feedback_{row.Index}"

    feedback_options = ["Yes", "No"]
    fbIndex = f"fb_{row.Index}"
    st.session_state[fbIndex] = None
    selected_feedback = st.radio(
        "Is this recipe helpful?",
        feedback_options,
        index=None,
        key=fbIndex,
        on_change=on_click_extract_feedback_to_dict,
        args=(row, user_inputs, feedback_model)
    )

# if user clicks on "Submit feedback and/or find another Recipe" button
# process feedback and save it in a json file and refresh the screen
def save_feedback_update_wt_refresh_screen(row, user_inputs, feedback_model):
    if not st.session_state.get('feedback_dict'):
        st.warning("No feedback to save.")
    else:
        if "combined_name_ingredients" not in st.session_state["stored_recommendations"].columns:
            st.session_state["stored_recommendations"]["combined_name_ingredients"] = (
                st.session_state["stored_recommendations"]["name"].astype(str) + " " +
                st.session_state["stored_recommendations"]["cleaned_ingredients"].astype(str)
            )

        # Access feedback_dict from session state
        feedback_dict = st.session_state["feedback_dict"]

        # Filter recommendations based on feedback keys (row indices)
        stored_recommendations = st.session_state.get('stored_recommendations', None)

        if stored_recommendations is not None:
            recommendations_df = pd.DataFrame(stored_recommendations)

            # Add an index column to ensure correct mapping
            recommendations_df.reset_index(inplace=True)

            # Convert feedback indices to dish names
            feedback_with_names = {
                recommendations_df.loc[idx, "name"]: feedback
                for idx, feedback in st.session_state["feedback_dict"].items()
                if idx in recommendations_df.index
            }

            # Filter recommendations based on indices present in the feedback_dict
            filtered_recommendations = recommendations_df[recommendations_df['index'].isin(feedback_dict.keys())
                                        ][['name', 'similarity_score']].to_dict(orient="records") 
        else:
            filtered_recommendations = []

        # Construct the feedback data with only the relevant recommendation and feedback
        feedback_data = {
            "user_inputs": user_inputs,
            "feedback": feedback_with_names,
            "recommendations": filtered_recommendations
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

        print("Feedback Saved Successfully, please check user_feedback.json in models folder to check how feedback is saved")  # Debugging step