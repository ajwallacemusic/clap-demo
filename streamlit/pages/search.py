import streamlit as st
from menu import menu
from app import INDEX

def get_text_embedding(query):
    text_data = st.session_state.clap_tokenizer([query], padding=True, return_tensors="pt")
    text_embed = st.session_state.clap_model.get_text_features(**text_data)
    text_arr = text_embed.detach().numpy()[0]
    return text_arr

def vector_search(query):
    text_arr = get_text_embedding(query)

    # Search for the document.
    query_body = {
        'size': 10,
        'query': {
            'knn': {
                'audio_embedding': {
                    'k': 100,
                    'vector': text_arr
                }
            }
        }
    }
    response = st.session_state.search_client.search(
        body = query_body,
        index = INDEX
    )

    return response    

def bm25_search(query):
    query_body = {
        "size": 5,
        "query": {
            "multi_match": {
            "query": query,
            "type": "best_fields",
            "fields": [
                "audio_set",
                "title",
                "genres",
                "album",
                "artist",
                "track_id"
            ]
            }
        }
    }
    response = st.session_state.search_client.search(
        body = query_body,
        index = INDEX
    )

    return response    

def hybrid_search(query):
    text_arr = get_text_embedding(query)

    query_body = {
        "size": 10,
        "query": {
            "hybrid": {
            "queries": [
                {
                "bool": {
                    "should": [
                    {
                        "multi_match": {
                        "query": query,
                        "type": "best_fields",
                        "fields": [
                            "audio_set",
                            "title^1000",
                            "genres",
                            "album",
                            "artist",
                            "track_id^1000"
                        ]
                        }
                    }
                    ]
                }
                },
                {
                "knn": {
                    "audio_embedding": {
                    "k": 25,
                    "vector": text_arr
                    }
                }
                }
            ]
            }
        },
        "search_pipeline": {
            "phase_results_processors": [
                {
                "normalization-processor": {
                    "normalization": {
                    "technique": "min_max"
                    },
                    "combination": {
                    "technique": "arithmetic_mean",
                    "parameters": {
                        "weights": [
                        0.8,
                        0.2
                        ]
                    }
                    }
                }
                }
            ]
            }
    }
    
    params = {}
    # params['search_pipeline'] = 'hybrid-audio-search'
    response = st.session_state.search_client.search(
        body = query_body,
        index = INDEX
    )
    return response

def display_results(response):
    hits = response['hits']['hits']
    num_hits = response['hits']['total']['value']
    st.write(str(num_hits) + " Results")
    for hit in hits:
        with st.container(border=True):
            colTitle, colGenres, colScore = st.columns(3)
            if 'title' in hit['_source'] and 'genres' in hit['_source']:
                with colTitle:
                    st.write("Title: " + hit['_source']['title'])
                with colGenres:
                    genres = hit['_source']['genres']
                    st.markdown(genres)
            with colScore:
                st.write("Score: " + str(hit['_score']))
            if 'track_id' in hit['_source']:
                st.write("Track ID: " + hit['_source']['track_id'])
            filepath = hit['_source']['filepath']
            st.audio(filepath)

st.title("Search Audio")
menu()

col1, col2 = st.columns(2)

with col1:
    query = st.text_input("Enter a Search Term")

with col2:
    st.session_state.search_type = st.selectbox(
                "Choose a Search Type",
                ("Vector", "BM25", "Hybrid")
            )

if st.button("Search"):
    if st.session_state.search_type == "Vector":
        response = vector_search(query)
        display_results(response)
    elif st.session_state.search_type == "BM25":
        response = bm25_search(query)
        display_results(response)
    elif st.session_state.search_type == "Hybrid":
        response = hybrid_search(query)
        display_results(response)
