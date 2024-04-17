import streamlit as st
import pyaudio
from transformers import pipeline, ClapModel, ClapProcessor, AutoTokenizer
from opensearchpy import OpenSearch

from menu import menu

# payaudio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
INDEX = "clap"

def init_session_state():
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = []
        st.session_state.recording = False
        st.session_state.audio = None
        st.session_state.clap_model = None
        st.session_state.clap_processor = None
        st.session_state.clap_tokenizer = None
        st.session_state.clap_audio_classifier = None

def create_search_client():
    # create opensearch client
    host = 'localhost'
    port = 9200

    # Create the client with ssl and auth disabled, NOT to be used for production!
    client = OpenSearch(
        hosts = [{'host': host, 'port': port}],
        http_compress = True, # enables gzip compression for request bodies
        use_ssl = False,
        verify_certs = False,
        ssl_assert_hostname = False,
        ssl_show_warn = False,
    )
    st.session_state.search_client = client

def init_clap():
    # init ML processors/models/tokenizers
    st.session_state.clap_audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/larger_clap_music_and_speech")
    st.session_state.clap_model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
    st.session_state.clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
    st.session_state.clap_tokenizer = AutoTokenizer.from_pretrained("laion/larger_clap_music_and_speech")

def set_up_audio():
    # audio stream
    
    audio = pyaudio.PyAudio()

    # Enumerate available audio devices
    num_devices = audio.get_device_count()
    st.write(num_devices)
    devices = [audio.get_device_info_by_index(i) for i in range(num_devices)]
    device_names = [audio.get_device_info_by_index(i)['name'] for i in range(num_devices)]

    # Find the index of the desired device by name
    desired_input_device_name = st.selectbox("Select an input audio device", device_names)
    desired_input_device_index = None
    for i, device in enumerate(devices):
        if device['name'] == desired_input_device_name:
            desired_input_device_index = i
            break
    


    desired_output_device_name = st.selectbox("Select an output audio device", device_names)
    desired_output_device_index = None
    for i, device in enumerate(devices):
        if device['name'] == desired_output_device_name:
            desired_output_device_index = i
            break

    if st.button("Initialize Audio"):
        input_device = devices[desired_input_device_index]
        if input_device['name'] == 'Universal Audio Thunderbolt':
            format = pyaudio.paInt24
        else:
            format = FORMAT 
        input_stream = audio.open(format=format,
                            channels=CHANNELS,
                            rate=int(input_device['defaultSampleRate']),
                            input=True,
                            input_device_index=desired_input_device_index,
                            frames_per_buffer=CHUNK)
        
        output_device = devices[desired_output_device_index]
        if output_device['name'] == 'Universal Audio Thunderbolt':
            format = pyaudio.paInt24
        else:
            format = FORMAT 
        output_stream = audio.open(format=format,
                            channels=CHANNELS,
                            rate=int(output_device['defaultSampleRate']),
                            output=True,
                            output_device_index=desired_output_device_index,
                            frames_per_buffer=CHUNK)
        
        st.success("Audio Devices Initialized Succesfully")
        st.write("Input Device: " + input_device['name'])
        st.write("Output Device: " + output_device['name'])
        st.write(input_device)
        st.write(output_device['defaultSampleRate'])
        
        st.session_state.audio = audio
        st.session_state.stream = input_stream
        st.session_state.output_stream = output_stream

def terminate_audio():
    # close stream
    st.session_state.stream.stop_stream()
    st.session_state.stream.close()
    st.session_state.audio.terminate()

def main():
    st.title("CLAP Audio Search Project")
    desc = "This is a demo app showing how to implement audio search via the CLAP model"
    st.write(desc)

    # initialize the things
    init_session_state()
    menu()
    set_up_audio()
    init_clap()
    create_search_client()

    
if __name__ == "__main__":
    main()
