import pandas as pd
import numpy as np
import re
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Open a file to store the results
with open("output_results.txt", "w") as file:

    # Load datasets
    surveyor_df = pd.read_csv("surveyor_data.csv")
    garage_df = pd.read_csv("garage_data.csv")
    primary_parts_df = pd.read_csv("Primary_Parts_Code.csv")

    # Sample data for testing
    surveyor_df = surveyor_df.sample(n=2500, random_state=42)
    garage_df = garage_df.sample(n=2500, random_state=42)

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

    file.write("Merged Data Sample:\n")
    file.write(str(merged_df.head()) + "\n")

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
    file.write("Part Mapping using Fuzzy Matching:\n")
    file.write(str(similar_parts_df.head()) + "\n")

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
    file.write("Part Mapping using Cosine Similarity:\n")
    file.write(str(similar_parts_df_cosine.head()) + "\n")

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
        file.write("No frequent itemsets found. Try lowering the min_support threshold.\n")
    else:
        file.write("Frequent Itemsets:\n")
        file.write(str(frequent_itemsets.head()) + "\n")

        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
        file.write("Top Association Rules:\n")
        file.write(str(rules.head()) + "\n")

    # Aggregate TOTAL_AMOUNT for duplicate entries before pivoting
    aggregated_df = merged_df.groupby(['CLAIMNO', 'TXT_PARTS_NAME']).agg({'TOTAL_AMOUNT_y': 'sum'}).reset_index()

    # Create user-item interaction matrix using the aggregated data
    interaction_matrix = aggregated_df.pivot(index='CLAIMNO', columns='TXT_PARTS_NAME', values='TOTAL_AMOUNT_y').fillna(0)

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
    file.write("Recommendations using TruncatedSVD:\n")
    file.write(str(example_recommendation_svd) + "\n")

    # Anomaly detection using Isolation Forest
    anomaly_data = merged_df[["CLAIMNO", "TOTAL_AMOUNT_y"]].drop_duplicates()
    X = anomaly_data[["TOTAL_AMOUNT_y"]]
    iso_forest = IsolationForest(contamination=0.05)
    anomaly_data["Anomaly_Score"] = iso_forest.fit_predict(X)

    # Identify suspicious claims
    fraud_claims = anomaly_data[anomaly_data["Anomaly_Score"] == -1]
    file.write("Suspicious Claims Detected:\n")
    file.write(str(fraud_claims) + "\n")

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
    file.write("Most Commonly Damaged Parts:\n")
    file.write(str(common_parts_df.head(10)) + "\n")

    # Claim Distribution by Primary Parts
    primary_parts_counts = surveyor_df["NUM_PART_CODE"].value_counts()
    primary_parts_df = pd.DataFrame({
        "Primary Part Code": primary_parts_counts.index,
        "Claim Count": primary_parts_counts.values
    })
    file.write("Claim Distribution by Primary Parts:\n")
    file.write(str(primary_parts_df.head(10)) + "\n")

    # Top Secondary Parts for each Primary Part
    top_secondary_parts = {}
    for primary_part in primary_parts_df["Primary Part Code"]:
        secondary_parts = surveyor_df[surveyor_df["NUM_PART_CODE"] == primary_part]["TXT_PARTS_NAME"].value_counts().head(10)
        top_secondary_parts[primary_part] = secondary_parts

    file.write("Top Secondary Parts for each Primary Part:\n")
    for primary_part, secondary_parts in top_secondary_parts.items():
        file.write(f"Primary Part: {primary_part}\n")
        file.write(str(secondary_parts) + "\n")

    # Matplotlib/Seaborn plots can still be displayed as usual; however, since we can't render them to the file, we won't write them in the file.
