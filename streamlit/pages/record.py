import streamlit as st
import wave
import os
import librosa
from opensearchpy.helpers import bulk
from menu import menu
from app import CHUNK, FORMAT, RATE, INDEX

TO_INDEX_RECORDINGS_DIRECTORY = "./to_index_recordings/"
INDEXED_RECORDINGS_DIRECTORY = "./indexed_recordings"

def start_recording():
    st.session_state['recording'] = True    

def stop_recording():
    st.session_state['recording'] = False

def record_audio():
    with st.spinner("Recording..."):
        while st.session_state.recording:
            data = st.session_state.stream.read(CHUNK, exception_on_overflow = False)
            st.session_state.audio_data.append(data)

def save_audio(filename):
    if st.session_state.audio_data:
        wf = wave.open(f"{TO_INDEX_RECORDINGS_DIRECTORY}{filename}.wav", 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(st.session_state.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(st.session_state.audio_data))
        wf.close()
        st.success(f"saved {filename}.wav file to 'to index recordings' folder")
    else:
        st.write("No audio data recorded.")

def get_audio_embedding(filepath):
    y, sr = librosa.load(filepath)
    inputs = st.session_state.clap_processor(audios=y, return_tensors="pt", sampling_rate=48000)
    audio_embed = st.session_state.clap_model.get_audio_features(**inputs)
    arr = audio_embed.detach().numpy()[0]
    return arr

def bulk_index_documents(client, documents):
    actions = []
    for doc in documents:
        action = {
            "_index": INDEX,
            "_source": doc
        }
        actions.append(action)
    
    bulk(client, actions)

def bulk_index_audio():
    with st.spinner("Indexing..."):
        # List to store documents for bulk indexing
        bulk_docs = []
        # es bulk batch size
        batch_size = 100

        for filename in os.listdir(TO_INDEX_RECORDINGS_DIRECTORY):
            track_id = filename[:-4]

            if filename.endswith(".wav") or filename.endswith(".mp3"):
                write_filepath = os.path.join(TO_INDEX_RECORDINGS_DIRECTORY, filename)
                read_filepath = os.path.join(INDEXED_RECORDINGS_DIRECTORY, filename)
                print("Processing:", write_filepath)
                print("track_id: ", track_id)

                arr = get_audio_embedding(write_filepath)

                doc = {
                    "audio_embedding": arr,
                    "audio_set": "aj_local_recordings",
                    "track_id": track_id,
                    "filepath": read_filepath,
                }

                # Add document to bulk indexing list
                bulk_docs.append(doc)

                # move audio file
                os.rename(write_filepath, read_filepath)
                
                # Perform bulk indexing if batch size is reached
                if len(bulk_docs) == batch_size:
                    bulk_index_documents(bulk_docs)
                    bulk_docs = []
                    
            # Index any remaining documents
        if bulk_docs:
            bulk_index_documents(st.session_state.search_client, bulk_docs)

        st.session_state.bulk_index = False
        print("Bulk indexing completed.")


menu()
# page display
st.title("Record Audio")

filename = st.text_input("Enter a name for the recording", "recorded_audio")

# record if start button is pressed, stop recording when stop button is pressed
# index if index button is pressed
col1, col2, col3= st.columns(3)
start = col1.button('Record', on_click=start_recording)
stop = col2.button('Stop', on_click=stop_recording)
col3.button('Index', on_click=bulk_index_audio)

if start:
    st.session_state.audio_data = []
    record_audio()

if stop:
    save_audio(filename)

# display recordings that exist locally that have been indexed
st.header("Indexed Recordings")
indexed_recordings = os.path.join(INDEXED_RECORDINGS_DIRECTORY)

for recording in os.listdir(indexed_recordings):
    if recording.endswith(".wav") or recording.endswith(".mp3"):
        st.write(recording)
        st.audio(f"{indexed_recordings}/{recording}")

# display local recordings that haven't been indexed
st.header("Recordings To Index")
to_index_recordings = os.path.join(TO_INDEX_RECORDINGS_DIRECTORY)

for recording in os.listdir(to_index_recordings):
    st.write(recording)
    st.audio(f"{to_index_recordings}/{recording}")