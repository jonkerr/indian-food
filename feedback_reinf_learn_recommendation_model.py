import streamlit as st
import numpy as np
import os
import json
import pandas as pd
from recommendation_models import compare_recommendation_models

class FeedbackRecommendationModel:
    # initiating feedback model
    def __init__(self, recipe_data, feedback_file="models/user_feedback.json"):
        self.recipe_data = recipe_data
        self.feedback_file = feedback_file
        # Ensure the feedback file exists
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, "w") as f:
                json.dump([], f)

    # function to load feedback data to the json file "user_feedback.json" under models folder
    def load_feedback(self):
        """Load feedback data from the feedback file."""
        try:
            with open(self.feedback_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    # this function aggregates the feedback for the user selected options from feedback data
    def aggregate_feedback(self, recommendations, feedback_data, user_inputs):
        """
        Aggregate feedback across multiple entries for the same recipe ID 
        and match feedback based on user-given optional parameters.
        """
        # Filter feedback entries that match the current user inputs
        relevant_feedback = []
        for feedback_entry in feedback_data:
            entry_user_inputs = feedback_entry.get("user_inputs", {})
            if all(
                entry_user_inputs.get(key) == user_inputs.get(key)
                for key in user_inputs.keys()
                if user_inputs[key] != "Select an option"
            ):
                relevant_feedback.append(feedback_entry["feedback"])

        # Initialize weight adjustment for all recipe IDs
        recommendations["weight_adjustment"] = 0

        # Process relevant feedback and compute average adjustments
        adjustment_dict = {}
        for feedback_dict in relevant_feedback:
            for recipe_id, feedback in feedback_dict.items():
                if recipe_id not in adjustment_dict:
                    adjustment_dict[recipe_id] = []
                if feedback == "helpful":
                    adjustment_dict[recipe_id].append(1)
                elif feedback == "not_related":
                    adjustment_dict[recipe_id].append(-1)

        # Assign average weight adjustment for each recipe ID
        for recipe_id, adjustments in adjustment_dict.items():
            if recipe_id in recommendations.index:
                avg_adjustment = np.mean(adjustments)
                recommendations.loc[recipe_id, "weight_adjustment"] = avg_adjustment

        return recommendations

    # this function uses user feedback into account to check if there is feedback matching user preferences, 
    # if yes, then apply average weights of all matching feedback in addition to similarity score calculated by compare_recommendations() 
    # then use combined score for updated recommendations
    def update_weights_with_feedback(self, recommendations, user_inputs):
        """Update weights based on feedback."""
        feedback_data = self.load_feedback()

        # If no feedback exists, skip weight adjustments
        if not feedback_data:
            return recommendations

        # Ensure similarity_score is numeric
        recommendations["similarity_score"] = pd.to_numeric(
            recommendations["similarity_score"], errors="coerce"
        ).fillna(0)

        # Aggregate feedback adjustments and update recommendations
        recommendations = self.aggregate_feedback(recommendations, feedback_data, user_inputs)

        # Calculate the final score
        recommendations["final_score"] = recommendations["similarity_score"] + recommendations["weight_adjustment"]

        # Sort by the final score and remove duplicates
        recommendations = recommendations.sort_values(by="final_score", ascending=False)
        recommendations = recommendations.drop_duplicates(subset="name", keep="first").reset_index(drop=True)

        return recommendations
    
    # this function takes into account recommendatios created by compare_recommendations() functions which provides upto top 5 recipes 
    # ranked by nmf/svd tfidf/count models
    # then calls update_weights_with_feedback function which updates the weights of recommended recipes based on user feedback and suggests rerated recipes
    def get_weighted_recommendations(self, filtered_recipes, user_inputs):
        """Generate recommendations using compare_recommendation_models and adjust weights based on feedback."""
        # Step 1: Call compare_recommendation_models to get initial recommendations
        recommendations = compare_recommendation_models(filtered_recipes)

        # Debug: Print recipe names from recommendations
        if "name" in recommendations.columns:
            recipe_names = recommendations["name"].tolist()
            print("Recommended Recipe Names:", recipe_names)  # Debug print
        else:
            print("No 'name' column found in recommendations. Available columns:", recommendations.columns)

        # Step 2: If only one recommendation is provided, return it without feedback processing
        if len(recommendations) == 1:
            print('only 1 recommended recipe, displaying as is')
            return recommendations

        # Step 3: If no matching feedback exists, display recommendations as is
        feedback_data = self.load_feedback()
        if not feedback_data:
            print('no feedback for user inputs, displaying as is')
            return recommendations

        # Step 4: Update weights based on feedback and return recommendations
        recommendations = self.update_weights_with_feedback(recommendations, user_inputs)
        return recommendations