import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import spotify_music_recommender as smr

if "song_init" not in st.session_state:
    st.session_state.song_init = False


def song_page(name, year):
    song_uri = smr.find_song_uri(name, year)
    formatted_song_uri = song_uri.split(':')[-1]
    uri_link = f'https://open.spotify.com/embed/track/{formatted_song_uri}?utm_source=generator'
    components.iframe(uri_link, height=100)


def spr_sidebar():
    menu = option_menu(
        menu_title=None,
        options=['Home', 'Results', 'About'],
        icons=['house', 'book', 'info-square'],
        menu_icon='cast',
        default_index=0,
        orientation='horizontal'
    )
    if menu == 'Home':
        st.session_state.app_mode = 'Home'
    elif menu == 'Results':
        st.session_state.app_mode = 'Results'
    elif menu == 'About':
        st.session_state.app_mode = 'About'
    # elif menu == 'How It Works':
    #     st.session_state.app_mode = 'How It Works'


def home_page():

    # App layout
    st.title("Spotify Music Recommender")

    # Song input section
    # st.subheader("")
    col1, col2 = st.columns(2)
    song_input = col1.text_input("Enter a song:")
    year_input = col2.text_input("Enter the year:")

    # Button section
    # st.subheader("")
    col3, col4 = st.columns(2)
    find_song_button = col3.button("Find Song")
    find_random_song_button = col4.button("Random Song")

    # Critic input section
    st.subheader("Song Review")
    critic_input = st.text_input("")

    # Prediction button
    predict_button = st.button("Start Prediction")

    if find_song_button:
        song_page(song_input, year_input)
    elif find_random_song_button:
        find_random_song()
    elif song_input == "" and year_input == "" and not st.session_state.song_init:
        st.session_state.song_init = True
        find_random_song()

    if predict_button:
        with st.spinner('Getting Recommendations...'):
            try:
                data_path = "data/data.csv"
                file_path = "data/pipeline.pkl"
                cluster_labels = "data/cluster_labels.csv"
                song_cluster_pipeline, data, number_cols = smr.get_model_values(
                    data_path, file_path, cluster_labels)
                user_critic_text = critic_input
                rec_splitted = smr.get_recommendation_array(
                    song_input, year_input, number_cols, user_critic_text)
                res = smr.recommend_gpt(
                    rec_splitted, data, song_cluster_pipeline, 15)
                st.session_state.song_uris = smr.get_rec_song_uri(res)
                st.write("You can access recommended song at result page")
            except:
                st.write("An error occured please try again")


# def text_field(label, columns=None, **input_params):
#     c1, c2 = st.columns(columns or [1, 4])

#     # Display field name with some alignment
#     c1.markdown("##")
#     c1.markdown(label)

#     # Sets a default key parameter to avoid duplicate key errors
#     input_params.setdefault("key", label)

#     # Forward text input parameters
#     return c2.text_input("", **input_params)


def find_random_song():
    try:
        song_input, year_input = smr.get_random_song()
        song_page(song_input, year_input)
    except:
        song_input, year_input = "Heat Waves", "2020"
        song_page(song_input, year_input)


def result_page():
    if "song_uris" in st.session_state:
        for uri in st.session_state.song_uris:
            uri = uri.split(":")[-1]
            uri_link = "https://open.spotify.com/embed/track/" + \
                uri + "?utm_source=generator&theme=0"
            components.iframe(uri_link, height=80)
    else:
        st.subheader(
            "Please enter song informations and review then click start prediction")


def examples_page():
    st.subheader("Will added")


def About_page():
    st.header('Development')
    """
    Have you ever listened to a song and liked it overall, but felt that certain features could be improved? Maybe the chorus was too loud, the energy level wasn't quite right, or there were either too many or too few words. I've had those experiences too, and that's what inspired me to create a song recommender based on user reviews.

    The process is straightforward: simply type in the name of a song or choose a random one, and then enter your review. The recommender will analyze your review using ChatGPT and utilize the Spotify API to generate personalized song recommendations. It's an exciting way to enhance your music discovery and tailor the recommendations to your specific preferences.

    Although it's important to note that the dataset used for training the model was relatively small (170k), the recommender still aims to provide valuable suggestions. While it may not reach its full potential due to the limited data, it serves as a starting point for exploring new songs that align with your individual tastes.

    So, if you're looking to fine-tune your music experience and discover songs that better match your preferences, give this song recommender a try. Enter your review, and let the algorithm work its magic to recommend songs that you're more likely to enjoy.\n
    Github : [alanahmet](https://github.com/alanahmet) \n
    Mail : cs.ahmetyusufalan@gmail.com
    """
    st.subheader('Audio Features Explanation')
    """
    | Variable | Description |
    | :----: | :---: |
    | Acousticness | A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. |
    | Danceability | Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. |
    | Energy | Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. |
    | Instrumentalness | Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. |
    | Key | The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1. |
    | Liveness | Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. |
    | Loudness | The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db. |
    | Mode | Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0. |
    | Speechiness | Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. |
    | Tempo | The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. |
    | Time Signature | An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of "3/4", to "7/4". |
    | Valence | A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). |
    
    Information about features: [here](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)
    """

    st.subheader('Credit')
    """
    Thanks for base of streamlit application to [abdelrhmanelruby](https://github.com/abdelrhmanelruby/Spotify-Recommendation-System) and dataset can be found [here](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset) 
    """


def main():
    spr_sidebar()
    if st.session_state.app_mode == 'Home':
        home_page()
    if st.session_state.app_mode == 'Results':
        result_page()
    if st.session_state.app_mode == 'About':
        About_page()
    # if st.session_state.app_mode == 'How It Works':
    #     examples_page()


# Run main()
if __name__ == '__main__':
    main()
