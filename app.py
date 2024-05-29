import streamlit as st
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio
import scipy.io.wavfile as wavfile

# Initialize processor and model
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# Language dictionary for selection
lang_dict = {
    'English': 'eng',
    'Russian': 'rus',
    'Spanish': 'spa',
    # Add more languages as needed
}

# Streamlit app
st.title("Speech-to-Speech Translation")

# Language selection
source_lang = st.selectbox('Select Source Language', list(lang_dict.keys()))
target_lang = st.selectbox('Select Target Language', list(lang_dict.keys()))

# Input text or audio
input_type = st.radio("Select Input Type", ("Text", "Audio"))

if input_type == "Text":
    # Text input
    text_input = st.text_area("Enter Text to Translate")
    if st.button("Translate"):
        with st.spinner("Translating..."):
            text_inputs = processor(text=text_input, src_lang=lang_dict[source_lang], return_tensors="pt")
            translated_audio = model.generate(**text_inputs, tgt_lang=lang_dict[target_lang])[0].cpu().numpy().squeeze()
        st.audio(translated_audio, format='audio/wav')

else:
    # Audio input
    audio_file = st.file_uploader("Upload Audio File", type=["wav"])
    if audio_file:
        if st.button("Translate"):
            with st.spinner("Translating..."):
                audio, orig_freq = torchaudio.load(audio_file)
                audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000)
                audio_inputs = processor(audios=audio, return_tensors="pt")
                translated_audio = model.generate(**audio_inputs, tgt_lang=lang_dict[target_lang])[0].cpu().numpy().squeeze()
            st.audio(translated_audio, format='audio/wav', caption='Translated Audio')

            # Option to save translated audio
            if st.button("Save Translated Audio"):
                wavfile.write("translated_audio.wav", 16000, translated_audio)

