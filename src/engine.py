import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    def __init__(self, reduced_matrix, movies_df):
        self.reduced_matrix = reduced_matrix
        self.movies_df = movies_df
        
        # Pre-compute the cosine similarity matrix for all movies
        self.similarity_matrix = cosine_similarity(self.reduced_matrix)
        
        # Wrap it in a DataFrame for easy lookup by movie_id
        self.similarity_df = pd.DataFrame(
            self.similarity_matrix,
            index=self.reduced_matrix.index,
            columns=self.reduced_matrix.index
        )

    def get_recommendations(self, movie_title, top_n=5):
        """Finds the closest movies based on cosine similarity."""
        
        # 1. Find the movie ID by matching the title string (case-insensitive)
        match = self.movies_df[self.movies_df['title'].str.contains(movie_title, case=False, na=False)]

        if match.empty:
            return f"Error: Movie '{movie_title}' not found in the dataset."

        # Grab the first match
        movie_id = match.iloc[0]['movie_id']
        exact_title = match.iloc[0]['title']

        if movie_id not in self.similarity_df.index:
            return "Error: Movie ID not found in the similarity matrix."

        # 2. Get similarity scores for this specific movie and sort them
        similar_scores = self.similarity_df[movie_id].sort_values(ascending=False)

        # 3. Exclude the movie itself (index 0 will be the exact movie, correlation = 1.0)
        top_movie_ids = similar_scores.iloc[1:top_n+1].index

        # 4. Map the top IDs back to their actual string titles
        recommendations = self.movies_df[self.movies_df['movie_id'].isin(top_movie_ids)]['title'].tolist()

        return exact_title, recommendations