import numpy as np
import os
import json
import pandas as pd
from compare_recommendation_models import compare_recommendation_models

class FeedbackRecommendationModel:
    def __init__(self, recipe_data, feedback_file="models/user_feedback.json"):
        self.recipe_data = recipe_data
        self.feedback_file = feedback_file
        # Ensure the feedback file exists
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, "w") as f:
                json.dump([], f)

    def load_feedback(self):
        """Load feedback data from the feedback file."""
        try:
            with open(self.feedback_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

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

        # Process relevant feedback and compute average adjustments for recipe IDs
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
                # Filter recommendations to match recipe_id and optional parameters
                matching_recommendations = recommendations.loc[
                    (recommendations.index == recipe_id)
                ]

                # Further check optional parameters for alignment
                for key, value in user_inputs.items():
                    if value != "Select an option" and key in recommendations.columns:
                        matching_recommendations = matching_recommendations.loc[
                            matching_recommendations[key].str.lower() == value.lower()
                        ]

                # If matching entries are found, update the weight adjustment
                if not matching_recommendations.empty:
                    avg_adjustment = np.mean(adjustments)
                    recommendations.loc[matching_recommendations.index, "weight_adjustment"] = avg_adjustment

        return recommendations


    def update_weights_with_feedback(self, recommendations, user_inputs):
        """Update weights based on feedback."""
        feedback_data = self.load_feedback()

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

    def get_weighted_recommendations(self, filtered_recipes, user_inputs):
        """Generate recommendations using compare_recommendation_models and adjust weights based on feedback."""
        # Step 1: Call compare_recommendation_models to get initial recommendations
        recommendations = compare_recommendation_models(filtered_recipes)

        # Step 2: Update weights based on feedback
        recommendations = self.update_weights_with_feedback(recommendations, user_inputs)

        # Step 3: Return adjusted recommendations
        return recommendations