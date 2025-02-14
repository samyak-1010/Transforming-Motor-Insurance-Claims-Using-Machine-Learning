import pandas as pd
import numpy as np
import re
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.ensemble import IsolationForest

# Load datasets
surveyor_df = pd.read_csv("surveyor_data.csv")
garage_df = pd.read_csv("garage_data.csv")
primary_parts_df = pd.read_csv("Primary_Parts_Code.csv")

# Sample data for testing
surveyor_df = surveyor_df.sample(n=1000, random_state=42)
garage_df = garage_df.sample(n=1000, random_state=42)

# Convert text to lowercase & remove special characters
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

surveyor_df["TXT_PARTS_NAME"] = surveyor_df["TXT_PARTS_NAME"].apply(clean_text)
garage_df["PARTDESCRIPTION"] = garage_df["PARTDESCRIPTION"].apply(clean_text)

# Convert columns to string type for consistent merging
surveyor_df["REFERENCE_NUM"] = surveyor_df["REFERENCE_NUM"].astype(str)
surveyor_df["VEHICLE_MODEL_CODE"] = surveyor_df["VEHICLE_MODEL_CODE"].astype(str)
surveyor_df["CLAIMNO"] = surveyor_df["CLAIMNO"].astype(str)

garage_df["REFERENCE_NUM"] = garage_df["REFERENCE_NUM"].astype(str)
garage_df["VEHICLE_MODEL_CODE"] = garage_df["VEHICLE_MODEL_CODE"].astype(str)
garage_df["CLAIMNO"] = garage_df["CLAIMNO"].astype(str)

# Merge datasets on common fields
merged_df = pd.merge(surveyor_df, garage_df, on=["REFERENCE_NUM", "VEHICLE_MODEL_CODE", "CLAIMNO"], how="inner")

print("Merged Data Sample:")
print(merged_df.head())

# Function to find similar parts using fuzzy matching
def find_similar_parts(part_name, part_list, threshold=85):
    match, score = process.extractOne(part_name, part_list)
    return match if score >= threshold else None

# Get unique part names
survey_parts = surveyor_df["TXT_PARTS_NAME"].unique()
garage_parts = garage_df["PARTDESCRIPTION"].unique()

# Mapping parts using fuzzy matching
similar_parts_mapping = {part: find_similar_parts(part, garage_parts) for part in survey_parts}

# Create a DataFrame for similar parts
similar_parts_df = pd.DataFrame(similar_parts_mapping.items(), columns=["Surveyor Part", "Garage Part"])
print("Part Mapping using Fuzzy Matching:")
print(similar_parts_df.head())

# Vectorize the combined part descriptions using TF-IDF
combined_parts = np.concatenate((survey_parts, garage_parts))
vectorizer = TfidfVectorizer().fit_transform(combined_parts)

# Compute cosine similarity matrix
cosine_sim_matrix = cosine_similarity(vectorizer)

# Function to find the most similar part using cosine similarity
def find_similar_part_cosine(part_index, sim_matrix, threshold=0.1):
    sim_scores = sim_matrix[part_index]
    most_similar_index = np.argsort(sim_scores)[-2]  # Exclude the part itself
    if sim_scores[most_similar_index] >= threshold:
        return most_similar_index
    return None

# Mapping parts using cosine similarity
similar_parts_mapping_cosine = {
    part: combined_parts[find_similar_part_cosine(idx, cosine_sim_matrix)]
    for idx, part in enumerate(survey_parts)
    if find_similar_part_cosine(idx, cosine_sim_matrix) is not None
}

# Create a DataFrame for similar parts using cosine similarity
similar_parts_df_cosine = pd.DataFrame(
    similar_parts_mapping_cosine.items(), columns=["Surveyor Part", "Garage Part"]
)
print("Part Mapping using Cosine Similarity:")
print(similar_parts_df_cosine.head())

# Prepare transaction dataset for association rule mining
transactions = merged_df.groupby("CLAIMNO")["TXT_PARTS_NAME"].apply(list)

# Convert transactions into a one-hot encoded DataFrame
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_apriori = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori Algorithm with a lower support threshold
frequent_itemsets = apriori(df_apriori, min_support=0.01, use_colnames=True)

if frequent_itemsets.empty:
    print("No frequent itemsets found. Try lowering the min_support threshold.")
else:
    print("Frequent Itemsets:")
    print(frequent_itemsets.head())

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    print("Top Association Rules:")
    print(rules.head())

# Create user-item interaction matrix using the correct column name
if 'TOTAL_AMOUNT_x' in merged_df.columns:
    interaction_matrix = merged_df.pivot(index='CLAIMNO', columns='TXT_PARTS_NAME', values='TOTAL_AMOUNT_x').fillna(0)
elif 'TOTAL_AMOUNT_y' in merged_df.columns:
    interaction_matrix = merged_df.pivot(index='CLAIMNO', columns='TXT_PARTS_NAME', values='TOTAL_AMOUNT_y').fillna(0)
else:
    raise KeyError("TOTAL_AMOUNT column not found in merged_df")

# Apply TruncatedSVD
svd = TruncatedSVD(n_components=4, random_state=42)
latent_matrix = svd.fit_transform(interaction_matrix)

# Compute cosine similarity on the latent matrix
similarity_matrix = cosine_similarity(latent_matrix)

# Function to recommend parts based on similarity
def recommend_parts_svd(claim_id, similarity_matrix, top_n=5):
    claim_idx = interaction_matrix.index.get_loc(claim_id)
    sim_scores = similarity_matrix[claim_idx]
    similar_indices = np.argsort(sim_scores)[::-1]
    recommendations = interaction_matrix.index[similar_indices].tolist()[:top_n]
    return recommendations

# Example recommendation
example_recommendation_svd = recommend_parts_svd(interaction_matrix.index[0], similarity_matrix)
print("Recommendations using TruncatedSVD:", example_recommendation_svd)

# Anomaly detection using Isolation Forest
anomaly_data = merged_df[["CLAIMNO", "TOTAL_AMOUNT_y"]].drop_duplicates()
X = anomaly_data[["TOTAL_AMOUNT_y"]]
iso_forest = IsolationForest(contamination=0.05)
anomaly_data["Anomaly_Score"] = iso_forest.fit_predict(X)

# Identify suspicious claims
fraud_claims = anomaly_data[anomaly_data["Anomaly_Score"] == -1]
print("Suspicious Claims Detected:", fraud_claims)

# Commonly Damaged Parts Analysis
part_counts = surveyor_df["TXT_PARTS_NAME"].value_counts()
total_parts = part_counts.sum()
part_percentages = (part_counts / total_parts) * 100

# Display the most commonly damaged parts
common_parts_df = pd.DataFrame({
    "Part Name": part_counts.index,
    "Count": part_counts.values,
    "Percentage": part_percentages.values
})
print("Most Commonly Damaged Parts:")
print(common_parts_df.head(10))

# Claim Distribution by Primary Parts
primary_parts_counts = surveyor_df["NUM_PART_CODE"].value_counts()
primary_parts_df = pd.DataFrame({
    "Primary Part Code": primary_parts_counts.index,
    "Claim Count": primary_parts_counts.values
})
print("Claim Distribution by Primary Parts:")
print(primary_parts_df.head(10))

# Top Secondary Parts for each Primary Part
top_secondary_parts = {}
for primary_part in primary_parts_df["Primary Part Code"]:
    secondary_parts = surveyor_df[surveyor_df["NUM_PART_CODE"] == primary_part]["TXT_PARTS_NAME"].value_counts().head(10)
    top_secondary_parts[primary_part] = secondary_parts

print("Top Secondary Parts for each Primary Part:")
for primary_part, secondary_parts in top_secondary_parts.items():
    print(f"Primary Part: {primary_part}")
    print(secondary_parts)
