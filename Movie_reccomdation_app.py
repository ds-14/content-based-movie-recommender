import pandas as pd
import joblib 
import streamlit as st

# Function to load data
def load_data(file_path):
    data = pd.read_csv(file_path + "/" + 'movie_data_for_app.csv')
    dataframe = pd.read_csv(file_path + "/" + 'movie_dataframe_for_app.csv')
    return data, dataframe

# Function to load models
def load_models(file_path):
    sig = joblib.load(file_path + "/" + 'sigmoid_kernel.pkl')
    tfv = joblib.load(file_path + "/" + 'tfidf_vectorizer.pkl')
    return sig, tfv

# Function to give recommendations
def give_recommendations(movie_title, model, data, dataframe):
    try:
        # Create indices series
        indices = pd.Series(data=data.index, index=data['original_title'])
        
        # Get the index of the movie
        idx = indices[movie_title]
        
        # Get similarity scores
        if hasattr(model, 'toarray'):  # Check if sparse matrix
            model_scores = list(enumerate(model[idx].toarray()[0]))
        else:
            model_scores = list(enumerate(model[idx]))
        
        # Sort by similarity score
        model_scores_sorted = sorted(model_scores, key=lambda x: x[1], reverse=True)
        
        # Get top 10 (excluding the movie itself)
        model_scores_10 = model_scores_sorted[1:11]
        
        # Get movie indices
        movie_indices_10 = [i[0] for i in model_scores_10]
        
        # Return movie titles
        return dataframe['original_title'].iloc[movie_indices_10].values
    
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return []

# Streamlit App Configuration
st.set_page_config(page_title="Simple Movie Recommendation", layout="centered")

# Add loading state
@st.cache_data
def load_all_data():
    file_path = r"C:\Users\DELL\OneDrive\Desktop\P4DS4D2\dump objects"
    data, dataframe = load_data(file_path)
    return data, dataframe

@st.cache_resource
def load_all_models():
    file_path = r"C:\Users\DELL\OneDrive\Desktop\P4DS4D2\dump objects"
    sig, tfv = load_models(file_path)
    return sig, tfv

# Load data and models
try:
    with st.spinner('Loading data and models...'):
        data, dataframe = load_all_data()
        sig, tfv = load_all_models()
    
    # App Title and Description
    st.title("🎬 Simple Movie Recommender")
    st.write("Find movies similar to your favourite one!")
    
    # Movie selection
    movie_list = sorted(data['original_title'].unique())
    selected_movie = st.selectbox("Select a movie:", movie_list)
    
    # Recommendation button
    if st.button("Get Recommendations"):
        if selected_movie:
            with st.spinner('Finding similar movies...'):
                recommendations = give_recommendations(selected_movie, sig, data, dataframe)
            
            if len(recommendations) > 0:
                st.subheader(f"Movies similar to: {selected_movie}")
                
                # Display recommendations
                for index, movie in enumerate(recommendations, 1):
                    st.write(f"{index}. {movie}")
            else:
                st.warning("No recommendations found for this movie.")
        else:
            st.warning("Please select a movie first.")
    
    # Footer
    st.markdown("---")
    st.markdown("*This app uses content-based filtering to recommend movies.*")

except FileNotFoundError as e:
    st.error(f"❌ Error: Could not find data files. Please check the file path.")
    st.error(f"Details: {str(e)}")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.error("Please make sure all required files are in the correct location.")