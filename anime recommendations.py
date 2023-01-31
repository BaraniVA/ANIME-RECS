import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

# loading data set
data = pd.read_csv('C:/Users/User/.ipynb_checkpoints/anime_with_synopsis.csv')

data['Score'].replace("Unknown", 0.0, inplace=True)

# Convert the score column to float
data["Score"] = data["Score"].astype(float)

# Save the changes back to the csv file
data.to_csv("C:/Users/User/.ipynb_checkpoints/anime_with_synopsis_updated.csv", index=False)

# display the data
print(data)

# Save the updated dataframe back to the csv file
data.to_csv("C:/Users/User/.ipynb_checkpoints/anime_with_synopsis_updated.csv", index=False)

# Preprocess the data
data = data.fillna(0)
encoder = OneHotEncoder(handle_unknown='ignore')
genre_encoded = encoder.fit_transform(data[['Genres']])

# Calculate the similarity score
similarity_score = cosine_similarity(genre_encoded)

# Get the top N recommendations
def recommend(anime_id,N):
    scores = similarity_score[anime_id]
    top_anime = scores.argsort()[-N:]
    return data["Name"].iloc[top_anime]

# Recommend the top anime titles for the anime with ID 0
print(recommend(anime_id ,N ))