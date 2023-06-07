import streamlit as st
import tensorflow.keras as keras
import librosa
import math
import numpy as np
import os
from werkzeug.utils import secure_filename


def process_input(audio_file):
    SAMPLE_RATE = 22050
    NUM_MFCC = 13
    N_FTT = 2048
    HOP_LENGTH = 512
    TRACK_DURATION = track_duration
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    NUM_SEGMENTS = 10

    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

    signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)

    for d in range(10):
        start = samples_per_segment * d
        finish = start + samples_per_segment

        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T

        return mfcc


def predict():
    genre_dict = {10: "Blues", 1: "Metal", 2: "Country", 0: "Disco", 4: "Classical", 5: "Jazz", 3: "Rock", 9: "Pop", 8: "Hiphop", 7: "Reggae"}

    st.title("Music Genre Classifier")

    file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if file is not None:
        uploads_dir = ""
        file_path = os.path.join(uploads_dir, secure_filename(file.name))
        with open(file_path, "wb") as f:
            f.write(file.read())

        model = keras.models.load_model("MGC.h5")
        prediction = model.predict(X_to_predict)
        predicted_index = np.argmax(prediction, axis=1)
        predicted_genre = genre_dict[int(predicted_index)]

        st.success(f"Predicted genre: {predicted_genre}")


if __name__ == "__main__":
    predict()
