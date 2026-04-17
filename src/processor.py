import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class MatrixProcessor:
    def __init__(self, n_components=50):
        # We reduce the feature space to 50 principal components
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def create_matrix(self, ratings):
        """Creates an Item-User pivot table."""
        # Rows = Movies, Columns = Users. 
        # This allows us to find similarities between movies based on user rating patterns.
        matrix = ratings.pivot(index='movie_id', columns='user_id', values='rating')
        
        # Fill missing ratings with 0 (Standard approach for sparse matrices before scaling)
        matrix = matrix.fillna(0)
        return matrix

    def apply_pca(self, matrix):
        """Scales the data and applies PCA."""
        # 1. Normalize the data (Mean centering is crucial for PCA)
        scaler = StandardScaler()
        normalized_matrix = scaler.fit_transform(matrix)

        # 2. Apply PCA to compress user preferences into latent features
        reduced_matrix = self.pca.fit_transform(normalized_matrix)

        # Return as a DataFrame to keep track of the movie_id index
        return pd.DataFrame(reduced_matrix, index=matrix.index)