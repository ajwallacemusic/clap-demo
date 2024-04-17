import streamlit as st
from transformers import pipeline
import librosa, os
from menu import menu
from pages.record import INDEXED_RECORDINGS_DIRECTORY
from pages.audio_to_audio_search import get_audiofile

def get_label_with_highest_score(input_list):
    # Initialize the highest score and the corresponding label
    highest_score = -1
    highest_label = None
    
    # Iterate through the input list
    for item in input_list:
        # Check if the current score is higher than the highest score
        if item['score'] > highest_score:
            highest_score = item['score']
            highest_label = item['label']
    
    # Return the label with the highest score
    return highest_label

def classify_audio(text1, text2, audiofile):
    y, sr = librosa.load(audiofile)
    audio_classifier = st.session_state.clap_audio_classifier
    output = audio_classifier(y, candidate_labels=[text1, text2])

    return get_label_with_highest_score(output)

# page display
st.title("Classify Audio")
menu()

col1, col2 = st.columns(2)
with col1:
    text1 = st.text_input("Label 1", "rock")
with col2:
    text2 = st.text_input("Label 2", "pop")

# render file dropdown and show file
audiofile = get_audiofile()
st.audio(audiofile)

# option to upload file
uploaded_file = st.file_uploader("Or upload one")

if uploaded_file is not None:
    source = uploaded_file
    st.audio(source)
else:
    source = audiofile


if st.button("Classify"):
    label = classify_audio(text1, text2, source)
    st.write(label)
