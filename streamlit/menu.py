import streamlit as st

def menu():
    st.sidebar.page_link("app.py", label="Home")
    st.sidebar.page_link("pages/search.py", label="Search Audio")
    st.sidebar.page_link("pages/record.py", label="Record Audio")
    st.sidebar.page_link("pages/classify.py", label="Classify Audio")
    st.sidebar.page_link("pages/audio_to_audio_search.py", label="Find Similar Audio")