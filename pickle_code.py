import pickle
import numpy as np
import pandas as pd
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
import re

# Function to clean text (you can adjust this function based on your actual cleaning needs)
def clean_text(text):
    # Remove special characters, extra spaces, and make text lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    text = text.strip().lower()  # Remove leading/trailing spaces and convert to lowercase
    return text

# Load Pickle Files
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Load models and mappings
vectorizer = load_pickle("/kaggle/input/biegrberg/tfidf_vectorizer.pkl")
svd_model = load_pickle("/kaggle/input/biegrberg/svd_model.pkl")
iso_forest_model = load_pickle("/kaggle/input/biegrberg/isolation_forest_model.pkl")
similar_parts_mapping = load_pickle("/kaggle/input/biegrberg/similar_parts_mapping.pkl")

# Load real data
merged_df = pd.read_csv('/kaggle/working/merged_data.csv')  # Actual claims data
garage_df = pd.read_csv('/kaggle/input/muthking/garage_data.csv')
surveyor_df = pd.read_csv('/kaggle/input/muthking/surveyor_data.csv')

# Clean the text in the specified columns
surveyor_df["TXT_PARTS_NAME"] = surveyor_df["TXT_PARTS_NAME"].apply(clean_text)
garage_df["PARTDESCRIPTION"] = garage_df["PARTDESCRIPTION"].apply(clean_text)

# Create parts lists for fuzzy matching
garage_parts = garage_df["PARTDESCRIPTION"].tolist()
survey_parts = surveyor_df["TXT_PARTS_NAME"].tolist()

print("Available CLAIMNO values:", merged_df['CLAIMNO'].tolist())

# User input functions
def get_part_from_user():
    return input("Enter the part name to find similar parts: ").strip().lower()

def get_claim_from_user():
    return input("Enter the CLAIMNO to get recommendations: ").strip()

# Function to find similar parts using fuzzy matching
def find_similar_parts(part_name):
    match, score = process.extractOne(part_name, garage_parts)
    if score >= 85:  # Can adjust threshold
        return match
    return "No similar part found"

# Function to recommend parts using TruncatedSVD (from claim)
def recommend_parts_svd(claim_id, similarity_matrix, top_n=5):
    if claim_id not in merged_df_agg['CLAIMNO'].values:
        return "CLAIMNO not found!"
    
    claim_idx = list(merged_df_agg['CLAIMNO']).index(claim_id)
    sim_scores = similarity_matrix[claim_idx]
    similar_indices = np.argsort(sim_scores)[::-1]
    recommendations = merged_df_agg['CLAIMNO'].iloc[similar_indices[:top_n]].tolist()
    return recommendations

# Function to detect anomalies (fraudulent claims)
def detect_anomalies(total_amount):
    anomaly_score = iso_forest_model.predict([[total_amount]])
    return "Suspicious" if anomaly_score == -1 else "Normal"

# Aggregate duplicate combinations (sum or mean, depending on your use case)
merged_df_agg = merged_df.groupby(['CLAIMNO', 'TXT_PARTS_NAME'], as_index=False)['TOTAL_AMOUNT_y'].sum()

# Create the interaction matrix
interaction_matrix = merged_df_agg.pivot(index='CLAIMNO', columns='TXT_PARTS_NAME', values='TOTAL_AMOUNT_y').fillna(0)

# Create a similarity matrix using SVD
expected_features = svd_model.feature_names_in_
interaction_matrix = interaction_matrix.reindex(columns=expected_features, fill_value=0)
latent_matrix = svd_model.transform(interaction_matrix)
similarity_matrix = cosine_similarity(latent_matrix)

# User interaction loop
while True:
    print("\nSelect an option:")
    print("1. Find similar part")
    print("2. Get recommendations for a claim")
    print("3. Check for fraudulent claim")
    print("4. Exit")
    
    choice = input("Enter choice (1/2/3/4): ").strip()
    
    if choice == '1':
        part_name = get_part_from_user()
        similar_part = find_similar_parts(part_name)
        print(f"Similar part found: {similar_part}")
    
    elif choice == '2':
        claim_id = get_claim_from_user()
        recommendations = recommend_parts_svd(claim_id, similarity_matrix)
        print(f"Recommended claims: {recommendations}")
    
    elif choice == '3':
        total_amount = float(input("Enter the total amount to check for anomalies: "))
        result = detect_anomalies(total_amount)
        print(f"Claim status: {result}")
    
    elif choice == '4':
        print("Exiting...")
        break
    
    else:
        print("Invalid choice. Please try again.")