import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import os

from utils.plot import show_spec, show_wave


def show(znt, fs):
    stft_dict = dict(n_fft=1024, hop_length=256, window="hann")

    fig, axes = plt.subplots(2, 1, figsize=(16, 4), tight_layout=True)

    show_wave(znt, fs, ax=axes[0], color='b')

    axes[0].tick_params(axis='x', bottom=False, labelbottom=False)
    axes[0].set_xlim([0, znt.shape[-1]/fs])
    axes[0].set_xlabel('')

    show_spec(znt, fs, stft_dict['n_fft'], ax=axes[1])
    return fig,axes


st.title("Decompose Harmonic and Percussive Components with librosa")

uploaded_file = st.sidebar.file_uploader("Choose a file",type=["wav","mp3"])

if uploaded_file is not None:
    st.markdown("## Input File")
    if "wav" in uploaded_file.name[-4:]:
        st.audio(uploaded_file)
    elif "mp3" in uploaded_file.name[-4:]:
        st.audio(uploaded_file,format="audio/mp3")
    wav,fs = sf.read(uploaded_file)
    wav = librosa.resample(wav.T,orig_sr=fs,target_sr=16000).T
    wav = np.sum(wav,axis=1)

    wav_len_sec = len(wav)//16000
    start_sec,end_sec = st.slider(
        "start and end",
        value=(0, wav_len_sec),
        min_value=0,
        max_value=wav_len_sec
    )

    st.pyplot(show(wav[start_sec*16000:end_sec*16000],16000)[0])

    if st.sidebar.button("analyze"):
        st.markdown("## Result")
        wav_harm,wav_perc = librosa.effects.hpss(wav)
        # virtualfile_harm = io.BytesIO()
        filepath="aaa"
        sf.write(filepath+"_harm.wav",wav_harm,16000)
        sf.write(filepath+"_perc.wav",wav_perc,16000)
        # wavfile.write(virtualfile_harm,data=wav_harm,rate=16000)
        st.audio(filepath+"_harm.wav")
        st.pyplot(show(wav_harm[start_sec*16000:end_sec*16000],16000)[0])
        st.audio(filepath+"_perc.wav")
        st.pyplot(show(wav_perc[start_sec*16000:end_sec*16000],16000)[0])

        if st.sidebar.button("refresh"):
            os.remove(filepath+"_harm.wav")
            os.remove(filepath+"_perc.wav")
    
        # with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        #     wavfile.write(f.name, fs, wav_harm)
        #     with open(f.name, "rb") as wav_file:
        #         st.audio(wav_file.read(), format="audio/wav")
        
        # with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        #     wavfile.write(f.name, fs, wav_perc)
        #     with open(f.name, "rb") as wav_file:
        #         st.audio(wav_file.read(), format="audio/wav")