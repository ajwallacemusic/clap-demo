# CLAP Demo App
## Semantic Search on Audio Sources
This demo app demonstrates how to implement the CLAP model to perform text to audio search, audio label classification and audio to audio search.

### Directories
- `opensearch_setup/`
    - contains a jupyter notebook that can be used to:
        - install dependencies
        - start opensearch cluster via docker-compose
        - create the "clap" index with the predefined `clap_mapping.json` file
        - process and bulk index 3 audio data sources (if audio data is already downloaded. More on that here.)
            - fma
            - vocal_imitations
            - FUSS
- `streamlit/`
    - contains the streamlit application files for running the demo.
    - to run the app:
        - it's recommended to use a virtual python environment. This project was tested with python version 3.9.6 using `PyEnv` to manage python versions and `venv` virtual environment in VS Code IDE.
        - once virtual environment is setup, install dependencies either through the `CLAP_notebook.ipynb`, or in the streamlit directory with `pip install -r requirements.txt`.
        - you will also need to install streamlit in the virtual environment by running `pip install streamlit`.
        - it's not required to pre-load data with the notebook, but you can read more about that here. 
        - it is required to have opensearch cluster running in docker with an index called "clap" -- this can easily be done in the notebook.
        - once dependcies are installed and opensearch is running, cd to the `streamlit` directory and run `streamlit run app.py` to start the app.
    
---

## Downloading Audio Data
The demo app is setup to accept audio data from 3 audio data sources:
- fma
- vocal_imitations
- FUSS

Adding this audio is not required, the app allows recording and indexing user audio from a local machine, but it's helpful for demo purposes to have a larger data set to play with.

These are all available from the [LAION audio set github audio data project](https://github.com/LAION-AI/audio-dataset) (and were used to help train the pretrained CLAP models).

Clone the repo, and simply run the download script for the audio sets (for example, the `download_FUSS.sh` script will download the FUSS audio set). Unzip the downloads and paste the contents into a directory called `audio_data` in the clap-demo directory (same level as `streamlit/` and `opensearch_setup/`.)

You should then be able to run the sections of the notebook for processing and bulk indexing these audio data sets.

---

For more information about CLAP, checkout the github page, the research article, or the hugging face model page.