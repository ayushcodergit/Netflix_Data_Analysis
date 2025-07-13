
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load cleaned data
df = pd.read_csv("cleaned_netflix_data.csv")



# Recommender Setup 
df['description'].fillna("", inplace=True)
df['listed_in'].fillna("", inplace=True)
df['combined_features'] = df['listed_in'] + " " + df['description']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
df = df.reset_index()
title_indices = pd.Series(df.index, index=df['title'].str.lower())

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = title_indices.get(title.lower())
    if idx is None:
        return ["Title not found."]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()




# Streamlit UI
st.set_page_config(page_title="Netflix Data Analysis", layout="wide")

st.title("🎬 Netflix Data Analysis & Recommender")
st.markdown("An interactive dashboard to explore Netflix data and get content recommendations.")

# Sidebar options
menu = st.sidebar.selectbox("Navigate", ["Overview", "EDA", "Visualizations", "Recommender", "Insights"])

# Overview
if menu == "Overview":
    st.header("📊 Project Overview")
    st.markdown("""
    - **Dataset**: Netflix Titles (movies + TV shows)
    - **Steps**: Data Cleaning → EDA → Visualizations → Recommender → Insights
    - **Tools**: Pandas, Seaborn, Scikit-learn, Streamlit
    """)
    st.dataframe(df.head(10))

# EDA 
elif menu == "EDA":
    st.header("📈 Exploratory Data Analysis")
    st.subheader("Content Type Count")
    st.write(df['type'].value_counts())

    st.subheader("Content Added Per Year")
    st.write(df['year_added'].value_counts().sort_index())

    st.subheader("Top 10 Countries")
    top_countries = df['country'].str.split(',').explode().str.strip().value_counts().head(10)
    st.write(top_countries)

    st.subheader("Top 10 Genres")
    top_genres = df['listed_in'].str.split(',').explode().str.strip().value_counts().head(10)
    st.write(top_genres)

# Visualizations

elif menu == "Visualizations":
    st.header("📊 Visual Insights")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    st.subheader("Content Type Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='type', palette='Set2', ax=ax1)
    st.pyplot(fig1)

    st.subheader("Titles Added Over the Years")
    fig2, ax2 = plt.subplots()
    df['year_added'].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax2)
    st.pyplot(fig2)

    st.subheader("Top 10 Countries")
    fig3, ax3 = plt.subplots()
    # Recalculate top_countries and top_genres for plotting
    top_countries = df['country'].str.split(',').explode().str.strip().value_counts().head(10)
    top_genres = df['listed_in'].str.split(',').explode().str.strip().value_counts().head(10)

    sns.barplot(x= top_countries.values, y= top_countries.index, palette='coolwarm', ax=ax3)
    st.pyplot(fig3)

    st.subheader("Top 10 Genres")
    fig4, ax4 = plt.subplots()
    top_genres.plot(kind='bar', color='teal', ax=ax4)
    st.pyplot(fig4)

# Recommender
elif menu == "Recommender":
    st.header("🎯 Netflix Title Recommender")
    movie_input = st.text_input("Enter a Movie/Show Title:")
    if st.button("Recommend"):
        recommendations = get_recommendations(movie_input)
        st.write("🔮 Recommendations:")
        for i, rec in enumerate(recommendations):
            st.write(f"{i+1}. {rec}")

# Insights
elif menu == "Insights":
    st.header("📌 Key Insights")
    st.markdown("""
    - 🎬 **Movies dominate** over TV Shows
    - 📈 **2018–2020**: Highest content addition
    - 🌍 **Top countries**: USA, India, UK
    - 🎭 **Popular genres**: Drama, International, Comedy
    - 🤖 **ML Recommendation** helps suggest similar titles based on description & genre
    """)
