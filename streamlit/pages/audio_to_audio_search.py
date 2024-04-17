import streamlit as st
from menu import menu
import os
from pages.record import get_audio_embedding, INDEXED_RECORDINGS_DIRECTORY
from pages.search import display_results
from app import INDEX

# define the audio to audio search function
def audio_to_audio_search(audiofile):
    # use the clap model to get the audio embedding
    embed = get_audio_embedding(audiofile)

    query_body = {
        'size': 10,
        'query': {
            'knn': {
                'audio_embedding': {
                    'k': 100,
                    'vector': embed[0]
                }
            }
        }
    }
    response = st.session_state.search_client.search(
        body = query_body,
        index = INDEX
    )

    return response   

def get_audiofile():
    indexed_recordings = os.path.join(INDEXED_RECORDINGS_DIRECTORY)
    audiofiles = []
    for file in os.listdir(indexed_recordings):
        if file.endswith(".wav") or file.endswith(".mp3"):
            audiofiles.append(file)
    file = st.selectbox("Select an audio file to search against", audiofiles)
    audiofile = f"{INDEXED_RECORDINGS_DIRECTORY}/{file}"
    return audiofile

#page display
menu()
st.title("Find Similar Audio")

audiofile = get_audiofile()
st.audio(audiofile)

uploaded_file = st.file_uploader("Or upload one")

if uploaded_file is not None:
    source = uploaded_file
    st.audio(source)
else:
    source = audiofile

if st.button("Search"):
    response = audio_to_audio_search(source)
    display_results(response)