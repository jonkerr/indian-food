import pandas as pd
import numpy as np
import re
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

processed_df = pd.read_pickle("data/processed_recipes.pkl")

#Find dish name in recipes and filter them based on user preferences
def filter_recipes(df, dish_name, cuisine=None, course=None, diet=None, prep_time=None, allergen_type=None, debug=False):
    # Escape special regex characters in the dish name
    escaped_dish_name = re.escape(dish_name)

    # Search for exact match of the joint name
    joint_name_pattern = rf'\b{escaped_dish_name}\b'
    filtered_df = df[df['name'].str.contains(joint_name_pattern, case=False, na=False, regex=True)]
    if debug:
        print(f"Number of dishes found with exact name '{dish_name}': {filtered_df.shape[0]}")
    
    # If no exact matches, search for exact match of individual words
    if filtered_df.empty:
        dish_name_parts = dish_name.split()
        word_patterns = [rf'\b{re.escape(word)}\b' for word in dish_name_parts]
        regex_pattern = '|'.join(word_patterns)  # Create regex to match any of the exact words
        filtered_df = df[df['name'].str.contains(regex_pattern, case=False, na=False, regex=True)]
        if debug:
            print(f"Number of dishes found with individual exact words '{dish_name_parts}': {filtered_df.shape[0]}")
    
    # If still no results, return None
    if filtered_df.empty:
        print("No recipes found matching the dish name criteria.")
        return None

    # Apply additional filters based on user preferences
    if cuisine:
        filtered_df = filtered_df[filtered_df['cuisine'].str.lower() == cuisine.lower()]
    if course:
        filtered_df = filtered_df[filtered_df['course'].str.lower() == course.lower()]
    if diet:
        filtered_df = filtered_df[filtered_df['diet'].str.lower() == diet.lower()]
    if prep_time:
        filtered_df = filtered_df[filtered_df['categorized_prep_time'] == prep_time]
    if allergen_type:
        # Exclude recipes with allergens in user preference
        allergen_set = set(allergen_type)
        filtered_df = filtered_df[~filtered_df['allergen_type'].apply(lambda x: bool(set(x) & allergen_set))]

    # Check the final filtered results
    if filtered_df.empty:
        print("No recipes found matching the full criteria.")
        return None
    
    if debug:
        print(f"Number of dishes after filtering: {filtered_df.shape[0]}")
    return filtered_df

#Calculate the best number of topics using coherence score for the filtered dataframe
def compute_coherence_scores(filtered_df, vectorizer, model, min_topics=2, max_topics_limit = 10):
    coherence_scores = []
    
    # Transform texts to TF-IDF matrix
    vectorizer_matrix = vectorizer.fit_transform(filtered_df["combined_name_ingredients"])
    
    # Tokenize each text in `texts` for Gensim's coherence calculation
    tokenized_texts = [text.split() for text in filtered_df["combined_name_ingredients"]]
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    max_topics = min(max_topics_limit, vectorizer_matrix.shape[0])
    num_topics_range = range(min_topics, max_topics + 1)
        
    # Iterate over each specified topic count
    for num_topics in num_topics_range :
        called_model = model(n_components=num_topics, random_state=42)
        called_matrix = called_model.fit_transform(vectorizer_matrix)
        
        # Extract top words for each topic and format as list of tokens
        feature_names = vectorizer.get_feature_names_out()
        topics = [[feature_names[i] for i in topic.argsort()[:-11:-1]] for topic in called_model.components_]
        
        tokenized_topics = [[word for word in topic] for topic in topics]# Coherence model requires a list of tokenized topics

        coherence_model = CoherenceModel(
            topics=tokenized_topics, 
            texts=tokenized_texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)
        
        #print(f"Number of Topics: {num_topics}, Coherence Score: {coherence_score}")

    # Find the best number of topics
    best_num_topics = num_topics_range[coherence_scores.index(max(coherence_scores))]
    print(f"Best Number of Topics: {best_num_topics} with Coherence Score: {max(coherence_scores)}")

    return best_num_topics

#Train the recommendation model using topic models and vectorizers
def get_recommendations(filtered_df, vectorizer, model, num_recommendations=5): 
    # Handle the case with only one dish in the filtered DataFrame
    if len(filtered_df) == 1:
        filtered_df = filtered_df.copy()
        filtered_df['similarity_score'] = ["100.00%"]  # Single dish is perfectly similar to itself
        #print("\nOnly one dish in the filtered DataFrame. Returning it as the recommendation.\n")
        return filtered_df[['name', 'similarity_score', 'cleaned_ingredients', 'cuisine', 
                            'course', 'diet', 'allergens', 'prep_time', 'instructions']]
    
    # Compute the optimal number of topics
    best_num_topics = compute_coherence_scores(filtered_df, vectorizer, model)

    # Transform the text data into a TF-IDF matrix
    vectorizer_matrix = vectorizer.fit_transform(filtered_df['combined_name_ingredients'])

    # Ensure all values in the TF-IDF matrix are non-negative and finite
    vectorizer_matrix = np.nan_to_num(vectorizer_matrix.toarray(), nan=0.0, posinf=0.0, neginf=0.0)

    # Train the NMF model with the optimal number of topics
    called_model = model(n_components=best_num_topics, random_state=42)
    called_model.fit(vectorizer_matrix)
    topic_matrix = called_model.transform(vectorizer_matrix)  # Topic distribution for each recipe

    # Copy the filtered DataFrame to avoid overwriting the original
    filtered_df = filtered_df.copy()

    # Add topic distributions to the filtered DataFrame
    for i in range(best_num_topics):
        filtered_df[f'topic_{i}'] = topic_matrix[:, i]

    # Compute cosine similarities for all dishes
    cosine_similarities = cosine_similarity(topic_matrix)

    if len(filtered_df) > num_recommendations:
        # More recipes than the number of recommendations
        highest_avg_score = 0.0
        best_set_indices = []

        # Loop through each dish and calculate the average similarity of its top similar dishes
        for dish_index in range(len(filtered_df)):
            # Get indices of the most similar recipes for the current dish
            similar_indices = cosine_similarities[dish_index].argsort()[-num_recommendations-1:-1][::-1]

            # Compute the average similarity score for the top similar dishes
            avg_similarity = cosine_similarities[dish_index][similar_indices].mean()

            # Update the best set of dishes if the current average similarity is higher
            if avg_similarity > highest_avg_score:
                highest_avg_score = avg_similarity
                best_set_indices = similar_indices
                best_similarity_scores = cosine_similarities[dish_index][similar_indices]

        # Create a DataFrame with the top recommended dishes
        recommended_recipes = filtered_df.iloc[best_set_indices].copy()
        recommended_recipes['similarity_score'] = [f"{(score * 100):.2f}%" for score in best_similarity_scores]

        # Print the highest average similarity score
        print(f"\nHighest average similarity score among all dishes: {highest_avg_score:.4f}\n")

        return recommended_recipes[['name', 'similarity_score', 'cleaned_ingredients', 'cuisine', 
                                    'course', 'diet', 'allergens', 'prep_time', 'instructions']]

    else:
        # Less than or equal to the number of recommendations
        base_recipe_index = 0  # Use the first recipe as the base
        similarity_scores = cosine_similarities[base_recipe_index]
        avg_similarity = similarity_scores.mean()

        # Add similarity scores to the DataFrame
        filtered_df['similarity_score'] = [f"{(score * 100):.2f}%" for score in similarity_scores]

        # Print the average similarity score
        print(f"\nAverage similarity score across all dishes: {avg_similarity:.4f}\n")

        return filtered_df[['name', 'similarity_score', 'cleaned_ingredients', 'cuisine', 
                            'course', 'diet', 'allergens', 'prep_time', 'instructions']]
    

def get_recommendations_nmf_tfidf(filtered_df): 
  return get_recommendations(filtered_df, TfidfVectorizer(), NMF, num_recommendations=5)

def get_recommendations_nmf_count(filtered_df): 
  return get_recommendations(filtered_df, CountVectorizer(), NMF, num_recommendations=5)

def get_recommendations_svd_tfidf(filtered_df):
  return get_recommendations(filtered_df, TfidfVectorizer(), TruncatedSVD, num_recommendations=5)

def get_recommendations_svd_count(filtered_df):
  return get_recommendations(filtered_df, CountVectorizer(), TruncatedSVD, num_recommendations=5)

#Compare recommendation models and choose the one with the highest average of similarity score as the best model 
def compare_recommendation_models(filtered_df):
    # Define a dictionary to store model functions and names
    model_functions = {"NMF_TFIDF": get_recommendations_nmf_tfidf, "NMF_Count": get_recommendations_nmf_count,
        "SVD_TFIDF": get_recommendations_svd_tfidf, "SVD_Count": get_recommendations_svd_count }

    results = []
    all_recommendations = {}

    for model_name, model_func in model_functions.items():
        # Get recommendations using the model
        recommended_recipes = model_func(filtered_df)

         # Store the recommendations for the current model
        all_recommendations[model_name] = recommended_recipes

        # Extract similarity scores and calculate the average
        similarity_scores = recommended_recipes['similarity_score'].str.rstrip('%').astype(float) / 100
        average_score = similarity_scores.mean()

        # Append the results to the list
        results.append({"Model": model_name, "Average Similarity Score": average_score})

    # Make a DataFrame for the results for better visualizacomparison
    similarity_df = pd.DataFrame(results)

    # Find the model with the highest average similarity score
    best_model = similarity_df.loc[similarity_df["Average Similarity Score"].idxmax()]
    best_model_name = best_model['Model']
    best_recommendations = all_recommendations[best_model_name]

    print("Comparison Table:")
    print(similarity_df)

    print(f"\nBest Model: {best_model['Model']} with Average Similarity Score: {best_model['Average Similarity Score']:.4f}")

    return best_recommendations

