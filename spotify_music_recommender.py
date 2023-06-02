import numpy as np
import pandas as pd
import openai
import spotipy
import pickle
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict


import warnings
warnings.filterwarnings("ignore")

def feature_get_pipeline_data_column_names():
    """
    Reads data from a CSV file, performs K-means clustering on numeric columns,
    and assigns cluster labels to the data.

    Returns:
    - song_cluster_pipeline: Pipeline object containing the scaler and K-means model.
    - data: DataFrame with the original data and cluster labels.
    - feature_column_names: List of column names containing numeric values.
    """
    
    data = pd.read_csv("data/data.csv")

    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                      ('kmeans', KMeans(n_clusters=20,
                                       verbose=False))
                                      ], verbose=False)

    X = data.select_dtypes(np.number)
    feature_column_names = list(X.columns)
    song_cluster_pipeline.fit(X)
    song_cluster_labels = song_cluster_pipeline.predict(X)
    data['cluster_label'] = song_cluster_labels

    return song_cluster_pipeline, data, feature_column_names

def get_model_values(data_path, file_path, cluster_path):
    
    with open(file_path, 'rb') as file:
        loaded_pipeline = pickle.load(file)
    data = pd.read_csv(data_path)
    labels = pd.read_csv(cluster_path)
    data["cluster_label"] = labels["cluster_label"]
    feature_column_names = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
                   'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
    return loaded_pipeline, data, feature_column_names


def find_song(name, year):
    """
    Finds a song on Spotify based on the song name and year.

    Args:
    - name: Name of the song.
    - year: Year of the song.

    Returns:
    - DataFrame containing the song's data.
    """
    
    if os.path.isfile(".\secret_keys.py"):
        import secret_keys
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=secret_keys.client_id, client_secret=secret_keys.client_secret))
    else:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=os.environ.get("client_id"), client_secret=os.environ.get("client_secret")))

    song_data = defaultdict()
    results = sp.search(q='track: {} year: {}'.format(name, year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


def find_song_uri(name, year):
    """
    Finds the Spotify URI of a song based on the song name and year.

    Args:
    - name: Name of the song.
    - year: Year of the song.

    Returns:
    - Spotify URI of the song.
    """
    
    # Create a Spotify client object.
    if os.path.isfile(".\secret_keys.py"):
        import secret_keys
        client = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=secret_keys.client_id, client_secret=secret_keys.client_secret))
    else:
        client = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=os.environ.get("client_id"), client_secret=os.environ.get("client_secret")))
    results = client.search(q='track: {} year: {}'.format(name, year), limit=1)
    track = results['tracks']['items'][0]
    song_id = track['uri']
    return song_id


def get_response(text):
    """
    Retrieves a response using OpenAI's GPT-3 language model.

    Args:
    - input_text: The input text for the model.

    Returns:
    - Generated response as a string.
    """
    
    if os.path.isfile(".\secret_keys.py"):
        import secret_keys
        openai.api_key = secret_keys.openai_api_key
    else:
        openai.api_key = os.environ.get("openai_api_key")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=text,
        temperature=0.7,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].get("text")


def get_finetune_text(user_critic, list_song_data):
    init_text = "I want you to act as a song recommender. I will provide you songs data with following format future_columns=[ <valence>, <published_year>, <acousticness>, <danceability>, <duration_ms>, <energy>, <explicit>,<instrumentalness>, <key>, <liveness>, <loudness>, <mode>, <popularity>, <speechiness>, <tempo>] \
     values and user critic about the given song. And you will change given array values based on user critic and return result array. Do not write any explanations or other words, just return an array that include changes in future_columns\
    and here is the np.describe() values of future_columns  \n\
    valence	year	acousticness	danceability	duration_ms	energy	explicit	instrumentalness	key	liveness	loudness	mode	popularity	speechiness	tempo \n \
    count	170653	170653	170653	170653	170653	170653	170653	170653	170653	170653	170653	170653	170653	170653	170653 \n \
    mean	0.528587211	1976.787241	0.502114764	0.537395535	230948.3107	0.482388835	0.084575132	0.167009581	5.199844128	0.205838655	-11.46799004	0.706902311	31.43179434	0.098393262	116.8615896 \n \
    std	0.263171464	25.91785256	0.376031725	0.176137736	126118.4147	0.267645705	0.278249228	0.313474674	3.515093906	0.174804661	5.697942912	0.455184191	21.82661514	0.162740072	30.70853304 \n \
    min	0	1921	0	0	5108	0	0	0	0	0	-60	0	0	0	0 \n \
    25%	0.317	1956	0.102	0.415	169827	0.255	0	0	2	0.0988	-14.615	0	11	0.0349	93.421 \n \
    50%	0.54	1977	0.516	0.548	207467	0.471	0	0.000216	5	0.136	-10.58	1	33	0.045	114.729 \n \
    75%	0.747	1999	0.893	0.668	262400	0.703	0	0.102	8	0.261	-7.183	1	48	0.0756	135.537 \n \
    max	1	2020	0.996	0.988	5403500	1	1	1	11	1	3.855	1	100	0.97	243.507"

    # init_last = "\n\n start with only typing random  future_columns values in given range as a array"
    # user_critic_example = "\n \"user_critic=it was too old and loud but i like the energy\" "
    # example_features = "future_columns=[0.68, 1976, 0.78, 0.62, 230948.3, 0.44, 0.22, 0.43, 5.2, 0.27, -9.67, 1, 31, 0.19, 118.86]"
    # test_input = init_text + user_last + user_critic + example_features + user_critic_last
    
    user_critic_last = "your output will be future_columns=[ <valence>, <published_year>, <acousticness>, <danceability>, <duration_ms>, <energy>, <explicit>,<instrumentalness>, <key>, <liveness>, <loudness>, <mode>, <popularity>, <speechiness>, <tempo>]  format"
    user_last = "\n\n start with the adjust following future_columns based on user_critic. "
    features = "future_columns=" + list_song_data
    real_input = init_text + user_last + \
        user_critic + features + user_critic_last

    return real_input


def format_gpt_output(raw_recommendation_array):
    formatted = raw_recommendation_array[3:-1].split(",")
    list_song_data = [float(i) for i in formatted]
    return list_song_data

def format_song_string(song_data, feature_column_names):
    list_song_data = song_data[feature_column_names].values.tolist()[0]
    list_song_data = '[' + ', '.join([str(num)
                                     for num in list_song_data]) + ']'
    return list_song_data

def format_chatgpt_recommendations(song_list, spotify_data, song_cluster_pipeline, n_songs=15):
    """
    Recommends a song using OpenAI's GPT-3 language model.

    Args:
    - song_name: The name of the song.
    - song_year: The year of the song.

    Returns:
    - Recommended song as a list of string.
    """
    
    feature_column_names = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
                   'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

    metadata_cols = ['name', 'year', 'artists']
    song_center = np.array(song_list)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[feature_column_names])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = spotify_data.iloc[index]
    # rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

def get_recommendation_song_uri(res):
    song_spotipy_info = []
    for song in res:
        song_spotipy_info.append(find_song_uri(song["name"], song["year"]))
    return song_spotipy_info

def get_recommendation_array(song_name, song_year, feature_column_names, user_critic_text):
    song_data = find_song(song_name, song_year)
    list_song_data = format_song_string(song_data, feature_column_names)
    user_critic = "\n \"user_critic=" + user_critic_text
    recommendation = get_response(get_finetune_text(user_critic, list_song_data))
    raw_recommendation_array = format_gpt_output(recommendation)
    return raw_recommendation_array


def get_random_song():
    data = pd.read_csv("data/data.csv")
    sample = data.sample(n=1)
    return sample.name, sample.year

def control():
    # song_cluster_pipeline, data, feature_column_names = feature_get_pipeline_data_column_names()
    data_path = "data/data.csv"
    file_path = "data/pipeline.pkl"
    cluster_labels = "data/cluster_labels.csv"
    song_cluster_pipeline, data, feature_column_names = get_model_values(
        data_path, file_path, cluster_labels)

    user_critic_text = "it was dull and very loud"
    song_name = "Poem of a Killer"
    song_year = 2022
    raw_recommendation_array = get_recommendation_array(
        song_name, song_year, feature_column_names, user_critic_text)
    
    result = format_chatgpt_recommendations(raw_recommendation_array, data, song_cluster_pipeline)
    print(result, get_recommendation_song_uri(result))

if __name__ == "__main__":
    control()