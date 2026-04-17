import pandas as pd

class DataLoader:
    def __init__(self, ratings_path, movies_path):
        self.ratings_path = ratings_path
        self.movies_path = movies_path

    def load_data(self):
        """Loads the MovieLens ratings and movie titles."""
        
        # Load ratings (u.data is tab-separated)
        ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_csv(
            self.ratings_path, 
            sep='\t', 
            names=ratings_cols, 
            encoding='latin-1'
        )

        # Load movies (u.item is pipe-separated, we only need ID and title)
        movies_cols = ['movie_id', 'title']
        movies = pd.read_csv(
            self.movies_path, 
            sep='|', 
            names=movies_cols, 
            usecols=[0, 1], 
            encoding='latin-1'
        )

        return ratings, movies