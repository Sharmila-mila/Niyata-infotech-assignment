Task 2: Unsupervised Learning (Clustering)

Spotify Music Clustering Using Unsupervised Machine Learning

PROJECT OVERVIEW:

This project focuses on clustering Spotify songs into meaningful groups using unsupervised machine learning techniques. The songs are grouped based on audio features such as danceability, energy, loudness, tempo, acousticness, and duration.
The main objective of this project is to identify similar music patterns and group songs into clusters using machine learning algorithms.

DATASET INFORMATION:

Dataset Name: Spotify Tracks Dataset
Source: Kaggle
Dataset Type: Tabular Dataset
Number of Records Used: 80,000 samples
Number of Features: 10
Feature Names:danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms

Dataset Link:
https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db

CLUSTERING ALGORITHMS:

1. K-Means Clustering

K-Means clustering was used as the main clustering algorithm to group similar songs together.

The model generated 5 clusters:

Cluster 0
Cluster 1
Cluster 2
Cluster 3
Cluster 4

These cluster labels represent different groups of songs with similar audio characteristics.

2. DBSCAN

DBSCAN was used as an additional clustering algorithm to identify dense song groups and detect possible outliers.

FINDING OPTIMAL CLUSTERS:

The following methods were used:

Elbow Method
Silhouette Score

Silhouette Score was used to evaluate how well the songs were separated into clusters.

INFERENCE:

Inference is used to predict the category of new Spotify songs using the trained K-Means clustering model.
In this project, song audio features such as danceability, energy, tempo, loudness, acousticness, and valence are given as input to the model.
The trained model analyzes these feature patterns and predicts the most suitable cluster for the song. Based on the predicted cluster, the song is categorized into meaningful groups such as:

Dance Songs
Sad Emotional Songs
Love Melody Songs
Party Vibe Songs
Relax Acoustic Songs




