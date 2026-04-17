from src.loader import DataLoader
from src.processor import MatrixProcessor
from src.engine import RecommendationEngine

def main():
    print("Initializing Movie Recommendation Engine...")

    # 1. Load the dataset
    loader = DataLoader(ratings_path='data/ml-100k/u.data', movies_path='data/ml-100k/u.item')
    ratings, movies = loader.load_data()
    print(f"Loaded {len(ratings)} ratings and {len(movies)} movies.")

    # 2. Process the matrix and apply PCA
    processor = MatrixProcessor(n_components=50) # Compressing to 50 latent features
    user_item_matrix = processor.create_matrix(ratings)
    reduced_matrix = processor.apply_pca(user_item_matrix)
    print(f"PCA reduced matrix shape from {user_item_matrix.shape} to {reduced_matrix.shape}.")

    # 3. Initialize the Recommendation Engine
    engine = RecommendationEngine(reduced_matrix, movies)
    print("Similarity matrix computed. Ready for recommendations!\n")

    # 4. Interactive Command Line Interface
    print("-" * 50)
    while True:
        user_input = input("Enter a movie title (or 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        result = engine.get_recommendations(user_input, top_n=5)

        if isinstance(result, str):
            print(f"{result}")
        else:
            exact_title, recs = result
            print(f"\nBecause you liked '{exact_title}', we recommend:")
            for i, rec in enumerate(recs, 1):
                print(f"   {i}. {rec}")
        print("-" * 50)

if __name__ == "__main__":
    main()