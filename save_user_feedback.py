import os
from datetime import datetime

# File to save feedback
feedback_file = "user_feedback.json"  # File to store feedback

# Save feedback to file
def save_feedback(recipe_feedback, user_inputs, recommendations):
    # Load existing feedback if the file exists
    try:
        with open(feedback_file, "r") as f:
            feedback_data = json.load(f)
    except FileNotFoundError:
        feedback_data = []

    # Append the new feedback
    feedback_entry = {
        "recipe_feedback": recipe_feedback,
        "user_inputs": user_inputs,
        "recommendations": recommendations.to_dict(orient="records")
    }
    feedback_data.append(feedback_entry)

    # Save feedback back to the file
    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=4)