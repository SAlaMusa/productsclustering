import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import linear_kernel
import os
from PIL import Image
import numpy as np
import pandas as pd

def load_data():
    # Load the dataset
    dataset_path = 'products.csv'
    df = pd.read_csv(dataset_path)

    return df

def preprocess_text_data(df):
    # Preprocess text data
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['name'].fillna(''))

    return tfidf_matrix

def preprocess_image_data(image_folder):
    # Preprocess image data
    image_files = os.listdir(image_folder)
    image_features = []

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = Image.open(image_path)
        img_array = np.array(img).flatten()
        image_features.append(img_array)

    return np.array(image_features)

def main():
    st.title("Product Clustering App")

    # Load data
    df = load_data()

    # Sidebar options
    st.sidebar.header("Clustering Options")
    clustering_method = st.sidebar.radio("Choose Clustering Method", ["Text Similarity", "Image Similarity"])

    # clustering
    if clustering_method == "Text Similarity":
        tfidf_matrix = preprocess_text_data(df)
        num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)

        # Perform clustering on text data
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(tfidf_matrix)

    elif clustering_method == "Image Similarity":
        image_folder = 'images'  # Folder containing product images
        image_features = preprocess_image_data(image_folder)
        num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)

        # Perform clustering on image data
        kmeans_image = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans_image.fit_predict(image_features)

    # Display clusters
    st.subheader("Product Clusters")
    for cluster in range(num_clusters):
        st.write(f"Cluster {cluster + 1}")
        cluster_data = df[df['cluster'] == cluster]
        for index, row in cluster_data.iterrows():
            # st.image(f"images/{row['image']}", caption=row['name'], use_column_width=True)
            st.image(row['image'], caption=row['name'], use_column_width=True)

            st.write(row['name'])
            st.write(f"Product Name: {row['name']}")
            st.write(f"Main Category: {row['main_category']}")
           
            st.write(f"Ratings: {row['ratings']}")
            # st.write(f"No. of Ratings: {row['no_of_ratings']}")
            # st.write(f"Discount Price: {row['discount_price']}")
            # st.write(f"Actual Price: {row['actual_price']}")

if __name__ == "__main__":
    main()
