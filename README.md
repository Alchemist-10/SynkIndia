# SynkIndia

![SynkIndia landing page](screenshots/landing%20page.jpg)

SynkIndia is a Streamlit app for translating an English source MP4 video into a selected Indian language while running a viseme-alignment / lip-sync style generation pipeline.

## Features

- Upload an English MP4 video.
- Choose a target Indian language: Hindi, Telugu, Tamil, Bengali, or Marathi.
- Adjust the number of fine-tuning epochs.
- Run the end-to-end translation pipeline and preview the generated output video.

## Project Structure

- `app.py` - Streamlit UI entry point.
- `src/audio_utils.py` - audio processing and translation helpers.
- `src/dataset.py` - multimodal dataset preparation.
- `src/model.py` - lip-sync autoencoder model.
- `src/pipeline.py` - end-to-end translation and rendering pipeline.

## Requirements

- Python 3.10+ recommended.
- `streamlit`
- `torch`
- `opencv-python`
- `ffmpeg` available on your system PATH

## Setup

1. Create and activate a virtual environment.
2. Install the project dependencies.
3. Make sure `ffmpeg` is installed and accessible from the command line.

Example:

```bash
pip install streamlit torch opencv-python
```

## Run

Start the app from the project root:

```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser.
2. Upload an English MP4 video.
3. Select the target language.
4. Set the number of epochs.
5. Click the execute button to generate the translated output.

## Notes

- The pipeline writes intermediate and final outputs under the local `data/` and `outputs/` folders.
- The generated video is returned by the Streamlit app after processing completes.
