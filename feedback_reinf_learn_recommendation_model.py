import streamlit as st
import numpy as np
import os
import json
import pandas as pd
from recommendation_models import compare_recommendation_models

# **** Overall Summary of this file **** 
# *****************************************
# 1) FeedbackRecommendationModel Class: It manages user feedback for recipe recommendations 
#          and adjusts recommendation weights based on feedback. It processes stored feedback 
#          and integrates it with similarity scores to generate weighted recipe recommendations.
# 2) __init__(): It Initializes the feedback model by setting up the recipe data and feedback file. 
#               This also Ensures the feedback file exists or creates it if missing.
# 3) load_feedback(): Loads feedback data from the specified feedback JSON file. 
#                     And then Returns an empty list if the file is not found.
# 4) aggregate_feedback(): Aggregates feedback for recipes based on user inputs, adjusting weight scores 
#                       for each recipe using feedback data. It standardizes recipe names for matching.
# 5) update_weights_with_feedback(): Updates the recommendation weights by combining similarity scores 
#                       and feedback weights. Sorts recipes by the final weighted score and returns
#                       updated recommendations.
# 6) get_weighted_recommendations(): Generates recipe recommendations using a comparison model and 
#           adjusts weights based on user feedback. Handles cases where feedback is unavailable or only 
#           one recommendation exists. Prints recipe order before reranking and after reranking.

class FeedbackRecommendationModel:
    # initiating feedback model with recommended recipe data and feedback file 
    # Also create a models/user_feedback.json file if it doesnt exists
    def __init__(self, recipe_data, feedback_file="models/user_feedback.json"):
        self.recipe_data = recipe_data
        self.feedback_file = feedback_file
        # checking if the feedback file exists, if not create a file 
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, "w") as f:
                json.dump([], f)

    # function to load feedback data from the json file "models/user_feedback.json"
    def load_feedback(self):
        """Load feedback data from the feedback file."""
        try:
            with open(self.feedback_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    # this function Aggregates feedback for recipes based on user inputs, adjusting weight scores 
    # for each recipe using feedback data. It standardizes recipe names for matching.
    def aggregate_feedback(self, recommendations, feedback_data, user_inputs):
        """
        Aggregate feedback across multiple entries for the same recipe name 
        and match feedback based on user-given optional parameters as well.
        """
        # Filter feedback entries that match the user inputs
        relevant_feedback = []
        relevant_feedback = [
            feedback_entry["feedback"]
            for feedback_entry in feedback_data
            if feedback_entry["user_inputs"] == user_inputs]

        # Initializing weight adjustment for all recipe IDs/name
        recommendations["weight_adjustment"] = 0

        # Processing relevant feedback 
        for feedback_dict in relevant_feedback:
            for recipe_name, feedback in feedback_dict.items():
                recipe_name = recipe_name.strip().lower()  # Standardize for matching
                if recipe_name in recommendations["name"].str.lower().values:
                    if feedback == "Yes":
                        recommendations.loc[recommendations["name"].str.lower() == recipe_name, "weight_adjustment"] += 1
                    elif feedback == "No":
                        recommendations.loc[recommendations["name"].str.lower() == recipe_name, "weight_adjustment"] -= 1

        return recommendations

    # this function Updates the recommendation weights by combining similarity scores and feedback weights.
    # Sorts recipes by the final weighted score and returns updated recommendations.
    def update_weights_with_feedback(self, recommendations, user_inputs):
        """Update weights based on feedback."""
        feedback_data = self.load_feedback()

        # If no feedback exists, skip weight adjustments
        if not feedback_data:
            return recommendations

        # Remove % and convert similarity_score to float
        recommendations["similarity_score"] = recommendations["similarity_score"].str.rstrip("%").astype(float) / 100

        # Aggregate feedback adjustments and get weights using feedback data
        recommendations = self.aggregate_feedback(recommendations, feedback_data, user_inputs)

        # Calculate the final score
        recommendations["final_score"] = recommendations["similarity_score"] * 0.7 + recommendations["weight_adjustment"] * 0.3
        # Sort by the final score
        recommendations = recommendations.sort_values(by="final_score", ascending=False)
        recommendations = recommendations.drop_duplicates(subset="name", keep="first").reset_index(drop=True)

        return recommendations
    
    # this function takes into account recommendatios created by compare_recommendations() functions 
    # which provides upto top 5 recipes ranked by nmf/svd tfidf/count models
    # then calls update_weights_with_feedback function which updates the weights of recommended recipes 
    # based on user feedback and suggests re-ranked recipes based on final score
    # Handles cases where feedback is unavailable or only one recommendation exists. 
    # Prints recipe order before reranking and after reranking
    def get_weighted_recommendations(self, filtered_recipes, user_inputs):
        """Generate recommendations using compare_recommendation_models and adjust weights based on feedback."""
        
        # Step 0: check combined_name_ingredients column exist in filtered_recipes df to deal with error
        if "combined_name_ingredients" not in filtered_recipes.columns:
            filtered_recipes["combined_name_ingredients"] = (
                filtered_recipes["name"].astype(str) + " " + 
                filtered_recipes["cleaned_ingredients"].astype(str)
            )
        # Step 1: Call compare_recommendation_models to get initial recommendations
        recommendations = compare_recommendation_models(filtered_recipes)
        print("Recommended Recipes by NLP based recommendation model with Similarity Scores:", recommendations[['name', 'similarity_score']])

        # Step 2: If only one recommendation is provided, return it without feedback processing
        if len(recommendations) == 1:
            print('only 1 recommended recipe, displaying as is')
            return recommendations

        # Step 3: If no matching feedback exists, display recommendations as is
        feedback_data = self.load_feedback()
        if not feedback_data:
            print('no feedback for user inputs, displaying as is')
            return recommendations

        # Step 4: Update weights based on existing feedback and return recommendations
        recommendations = self.update_weights_with_feedback(recommendations, user_inputs)
        print("Recommendations with Final weighted Scores after feedback model has been applied:", recommendations[['name', 'final_score']])

        return recommendations