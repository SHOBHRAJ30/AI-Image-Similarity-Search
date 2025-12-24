import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load saved features
features = np.load("features.npy")
image_names = np.load("images.npy")

# Change this index to test different shoes
query_index = 0   # 0 = first image, 1 = second image, etc.

query_feature = features[query_index]

similarity_scores = cosine_similarity([query_feature], features)[0]

# Sort by similarity (high to low), skip the same image
top_matches = similarity_scores.argsort()[::-1][1:6]

print("🔍 Query Image:", image_names[query_index])
print("\nSimilar Images:")
for idx in top_matches:
    print(image_names[idx], "Similarity:", round(similarity_scores[idx], 3))