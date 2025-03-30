import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class MovieRecommendationSystem:
    def __init__(self, movies_data_path):
        """Initialize the movie recommendation system with a dataset path."""
        # Load the movie dataset
        self.movies_df = pd.read_csv(movies_data_path)
        # Create a sample dataset if the file doesn't exist
        if 'sample' in movies_data_path:
            self.create_sample_dataset()
        # Preprocess the data
        self.preprocess_data()
        # Build the recommendation model
        self.build_model()
        
    def create_sample_dataset(self):
        """Create a sample movie dataset if no real dataset is provided."""
        # Define sample data
        data = {
            'movieId': list(range(1, 51)),
            'title': [
                'The Shawshank Redemption (1994)', 'The Godfather (1972)', 'The Dark Knight (2008)', 
                'The Godfather: Part II (1974)', 'Pulp Fiction (1994)', 'Schindler\'s List (1993)',
                'The Lord of the Rings: The Return of the King (2003)', '12 Angry Men (1957)',
                'The Lord of the Rings: The Fellowship of the Ring (2001)', 'Forrest Gump (1994)',
                'Inception (2010)', 'Fight Club (1999)', 'The Matrix (1999)', 'Goodfellas (1990)',
                'One Flew Over the Cuckoo\'s Nest (1975)', 'Se7en (1995)', 'It\'s a Wonderful Life (1946)',
                'The Silence of the Lambs (1991)', 'Saving Private Ryan (1998)', 'Interstellar (2014)',
                'The Green Mile (1999)', 'The Prestige (2006)', 'The Departed (2006)', 'Whiplash (2014)',
                'Gladiator (2000)', 'The Usual Suspects (1995)', 'American Beauty (1999)', 'Alien (1979)',
                'The Lion King (1994)', 'Back to the Future (1985)', 'The Shining (1980)', 'Django Unchained (2012)',
                'Finding Nemo (2003)', 'The Avengers (2012)', 'Inglourious Basterds (2009)', 'Toy Story (1995)',
                'Good Will Hunting (1997)', 'A Clockwork Orange (1971)', 'The Sixth Sense (1999)', 'Scarface (1983)',
                'Titanic (1997)', 'Full Metal Jacket (1987)', 'Monty Python and the Holy Grail (1975)',
                'The Truman Show (1998)', 'Jurassic Park (1993)', 'The Big Lebowski (1998)', 'No Country for Old Men (2007)',
                'Eternal Sunshine of the Spotless Mind (2004)', 'Blade Runner (1982)', 'Casablanca (1942)'
            ],
            'genres': [
                'Drama', 'Crime|Drama', 'Action|Crime|Drama', 'Crime|Drama', 'Crime|Drama', 'Biography|Drama|History',
                'Action|Adventure|Drama', 'Crime|Drama', 'Action|Adventure|Drama', 'Drama|Romance',
                'Action|Adventure|Sci-Fi', 'Drama', 'Action|Sci-Fi', 'Biography|Crime|Drama',
                'Drama', 'Crime|Drama|Mystery', 'Drama|Family|Fantasy', 'Crime|Drama|Thriller',
                'Drama|War', 'Adventure|Drama|Sci-Fi', 'Crime|Drama|Fantasy', 'Drama|Mystery|Sci-Fi',
                'Crime|Drama|Thriller', 'Drama|Music', 'Action|Adventure|Drama', 'Crime|Mystery|Thriller',
                'Drama', 'Horror|Sci-Fi', 'Animation|Adventure|Drama', 'Adventure|Comedy|Sci-Fi',
                'Drama|Horror', 'Drama|Western', 'Animation|Adventure|Comedy', 'Action|Adventure|Sci-Fi',
                'Adventure|Drama|War', 'Animation|Adventure|Comedy', 'Drama|Romance', 'Crime|Drama|Sci-Fi',
                'Drama|Mystery|Thriller', 'Crime|Drama', 'Drama|Romance', 'Drama|War', 'Adventure|Comedy|Fantasy',
                'Comedy|Drama', 'Action|Adventure|Sci-Fi', 'Comedy|Crime', 'Crime|Drama|Thriller',
                'Drama|Romance|Sci-Fi', 'Action|Drama|Sci-Fi', 'Drama|Romance|War'
            ],
            'director': [
                'Frank Darabont', 'Francis Ford Coppola', 'Christopher Nolan', 'Francis Ford Coppola', 
                'Quentin Tarantino', 'Steven Spielberg', 'Peter Jackson', 'Sidney Lumet',
                'Peter Jackson', 'Robert Zemeckis', 'Christopher Nolan', 'David Fincher',
                'Lana Wachowski', 'Martin Scorsese', 'Milos Forman', 'David Fincher',
                'Frank Capra', 'Jonathan Demme', 'Steven Spielberg', 'Christopher Nolan',
                'Frank Darabont', 'Christopher Nolan', 'Martin Scorsese', 'Damien Chazelle',
                'Ridley Scott', 'Bryan Singer', 'Sam Mendes', 'Ridley Scott',
                'Roger Allers', 'Robert Zemeckis', 'Stanley Kubrick', 'Quentin Tarantino',
                'Andrew Stanton', 'Joss Whedon', 'Quentin Tarantino', 'John Lasseter',
                'Gus Van Sant', 'Stanley Kubrick', 'M. Night Shyamalan', 'Brian De Palma',
                'James Cameron', 'Stanley Kubrick', 'Terry Gilliam', 'Peter Weir',
                'Steven Spielberg', 'Joel Coen', 'Joel Coen', 'Michel Gondry',
                'Ridley Scott', 'Michael Curtiz'
            ],
            'actors': [
                'Tim Robbins, Morgan Freeman', 'Marlon Brando, Al Pacino', 'Christian Bale, Heath Ledger', 
                'Al Pacino, Robert De Niro', 'John Travolta, Uma Thurman', 'Liam Neeson, Ralph Fiennes',
                'Elijah Wood, Viggo Mortensen', 'Henry Fonda, Lee J. Cobb', 'Elijah Wood, Ian McKellen',
                'Tom Hanks, Robin Wright', 'Leonardo DiCaprio, Joseph Gordon-Levitt', 'Brad Pitt, Edward Norton',
                'Keanu Reeves, Laurence Fishburne', 'Robert De Niro, Ray Liotta', 'Jack Nicholson, Louise Fletcher',
                'Brad Pitt, Morgan Freeman', 'James Stewart, Donna Reed', 'Jodie Foster, Anthony Hopkins',
                'Tom Hanks, Matt Damon', 'Matthew McConaughey, Anne Hathaway', 'Tom Hanks, Michael Clarke Duncan',
                'Hugh Jackman, Christian Bale', 'Leonardo DiCaprio, Matt Damon', 'Miles Teller, J.K. Simmons',
                'Russell Crowe, Joaquin Phoenix', 'Kevin Spacey, Gabriel Byrne', 'Kevin Spacey, Annette Bening',
                'Sigourney Weaver, Tom Skerritt', 'Matthew Broderick, Jeremy Irons', 'Michael J. Fox, Christopher Lloyd',
                'Jack Nicholson, Shelley Duvall', 'Jamie Foxx, Christoph Waltz', 'Albert Brooks, Ellen DeGeneres',
                'Robert Downey Jr., Chris Evans', 'Brad Pitt, Diane Kruger', 'Tom Hanks, Tim Allen',
                'Robin Williams, Matt Damon', 'Malcolm McDowell, Patrick Magee', 'Bruce Willis, Haley Joel Osment',
                'Al Pacino, Michelle Pfeiffer', 'Leonardo DiCaprio, Kate Winslet', 'Matthew Modine, Vincent D\'Onofrio',
                'Graham Chapman, John Cleese', 'Jim Carrey, Laura Linney', 'Sam Neill, Laura Dern',
                'Jeff Bridges, John Goodman', 'Tommy Lee Jones, Javier Bardem', 'Jim Carrey, Kate Winslet',
                'Harrison Ford, Rutger Hauer', 'Humphrey Bogart, Ingrid Bergman'
            ]
        }
        self.movies_df = pd.DataFrame(data)
        
    def preprocess_data(self):
        """Preprocess the movie data for recommendation."""
        # Extract year from title
        self.movies_df['year'] = self.movies_df['title'].apply(lambda x: self.extract_year(x))
        self.movies_df['title'] = self.movies_df['title'].apply(lambda x: self.clean_title(x))
        
        # Create content string from various features
        self.movies_df['content'] = (
            self.movies_df['genres'] + ' ' + 
            self.movies_df['director'] + ' ' + 
            self.movies_df['actors'] + ' ' + 
            self.movies_df['year'].astype(str)
        )
        
    def extract_year(self, title):
        """Extract year from movie title."""
        match = re.search(r'\((\d{4})\)', title)
        if match:
            return match.group(1)
        return '0'
    
    def clean_title(self, title):
        """Remove year from movie title."""
        return re.sub(r'\s*\(\d{4}\)', '', title)
    
    def build_model(self):
        """Build the recommendation model using TF-IDF and cosine similarity."""
        # Initialize TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Construct the TF-IDF matrix
        tfidf_matrix = tfidf.fit_transform(self.movies_df['content'])
        
        # Compute the cosine similarity matrix
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Create a mapping of movie titles to indices
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title']).drop_duplicates()
    
    def get_recommendations(self, title, n=10):
        """
        Get movie recommendations based on movie title.
        
        Args:
            title (str): Movie title
            n (int): Number of recommendations to return
            
        Returns:
            list: List of recommended movie titles
        """
        # Get the index of the movie
        if title not in self.indices:
            closest_match = self.find_closest_match(title)
            if closest_match:
                print(f"Movie '{title}' not found. Using closest match: '{closest_match}'")
                title = closest_match
            else:
                return []
        
        idx = self.indices[title]
        
        # Get the similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the top n most similar movies
        sim_scores = sim_scores[1:n+1]
        
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the top n most similar movies
        return self.movies_df['title'].iloc[movie_indices].tolist()
    
    def find_closest_match(self, title):
        """Find the closest matching title in the dataset."""
        # Simple implementation for finding closest match
        for movie_title in self.indices.index:
            if title.lower() in movie_title.lower():
                return movie_title
        return None
    
    def get_recommendations_by_genre(self, genre, n=10):
        """
        Get movie recommendations based on genre.
        
        Args:
            genre (str): Movie genre
            n (int): Number of recommendations to return
            
        Returns:
            list: List of recommended movie titles
        """
        # Filter movies by genre
        genre_movies = self.movies_df[self.movies_df['genres'].str.contains(genre, case=False)]
        
        if genre_movies.empty:
            return []
        
        # Return the top n movies in that genre
        return genre_movies['title'].head(n).tolist()
    
    def get_recommendations_by_director(self, director, n=10):
        """
        Get movie recommendations based on director.
        
        Args:
            director (str): Director name
            n (int): Number of recommendations to return
            
        Returns:
            list: List of recommended movie titles
        """
        # Filter movies by director
        director_movies = self.movies_df[self.movies_df['director'].str.contains(director, case=False)]
        
        if director_movies.empty:
            return []
        
        # Return the top n movies by that director
        return director_movies['title'].head(n).tolist()
    
    def get_recommendations_by_actor(self, actor, n=10):
        """
        Get movie recommendations based on actor.
        
        Args:
            actor (str): Actor name
            n (int): Number of recommendations to return
            
        Returns:
            list: List of recommended movie titles
        """
        # Filter movies by actor
        actor_movies = self.movies_df[self.movies_df['actors'].str.contains(actor, case=False)]
        
        if actor_movies.empty:
            return []
        
        # Return the top n movies with that actor
        return actor_movies['title'].head(n).tolist()


# Example usage
def main():
    # Create a movie recommendation system
    recommender = MovieRecommendationSystem("sample_movies.csv")
    
    print("Movie Recommendation System")
    print("==========================")
    
    while True:
        print("\nOptions:")
        print("1. Recommend movies similar to a movie")
        print("2. Recommend movies by genre")
        print("3. Recommend movies by director")
        print("4. Recommend movies by actor")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            movie_title = input("Enter a movie title: ")
            recommendations = recommender.get_recommendations(movie_title)
            
            if recommendations:
                print(f"\nMovies similar to '{movie_title}':")
                for i, movie in enumerate(recommendations, 1):
                    print(f"{i}. {movie}")
            else:
                print(f"No recommendations found for '{movie_title}'.")
                
        elif choice == '2':
            genre = input("Enter a genre (e.g., Action, Drama, Comedy): ")
            recommendations = recommender.get_recommendations_by_genre(genre)
            
            if recommendations:
                print(f"\nTop movies in '{genre}' genre:")
                for i, movie in enumerate(recommendations, 1):
                    print(f"{i}. {movie}")
            else:
                print(f"No movies found in '{genre}' genre.")
                
        elif choice == '3':
            director = input("Enter a director name: ")
            recommendations = recommender.get_recommendations_by_director(director)
            
            if recommendations:
                print(f"\nMovies directed by '{director}':")
                for i, movie in enumerate(recommendations, 1):
                    print(f"{i}. {movie}")
            else:
                print(f"No movies found by director '{director}'.")
                
        elif choice == '4':
            actor = input("Enter an actor name: ")
            recommendations = recommender.get_recommendations_by_actor(actor)
            
            if recommendations:
                print(f"\nMovies featuring '{actor}':")
                for i, movie in enumerate(recommendations, 1):
                    print(f"{i}. {movie}")
            else:
                print(f"No movies found featuring '{actor}'.")
                
        elif choice == '5':
            print("Thank you for using the Movie Recommendation System!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()