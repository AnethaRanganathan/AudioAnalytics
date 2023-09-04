!pip install whisper
import whisper
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration,AutoTokenizer
import torch
import textwrap
import pandas as p
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import pytube
import subprocess

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

st.title("Speech to Text Summarization using Whisper and BART")

# Create file input Widget to enable user to input file

url = st.text_input('Enter the video link here:')


if url is not None:
    if st.button("Generate Summary",type='primary'):

#video = "https://www.youtube.com/watch?v=Y8Tko2YC5hA";
        data = pytube.YouTube(url)

        # Converting and downloading as 'MP4' file
        audio = data.streams.get_audio_only()
        data2 = audio.download()
        
        model = whisper.load_model("base")
	
        st.markdown(f'<h1 style="font-size:16px;">Total Mins of the Video: {+round(get_length(data2)/60,2)} Mins</h1>', unsafe_allow_html=True)
        # load the entire audio file
        audio = whisper.load_audio(data2)

        options = {
            "language": "en", # input language, if omitted is auto detected
            "task": "transcribe",
            "fp16" : False # or "transcribe" if you just want transcription
        }
        result = whisper.transcribe(model, audio, **options)
        words = result["text"].split(" ")
        st.markdown(f'<h1 style="font-size:16px;">Number of Words in Video: {+len(words)}</h1>', unsafe_allow_html=True)
     #   print(result["text"])
            #begin1=time.time()
             # model and tokensizer defined 
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

                    # tokenize without truncation
        inputs_no_trunc = tokenizer(result["text"], max_length=None, return_tensors='pt', truncation=False)

                    # get batches of tokens corresponding to the exact model_max_length
        chunk_start = 0
        chunk_end = tokenizer.model_max_length  # == 1024 for Bart
        inputs_batch_lst = []
        while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
            inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]  # get batch of n tokens
            inputs_batch = torch.unsqueeze(inputs_batch, 0)
            inputs_batch_lst.append(inputs_batch)
            chunk_start += tokenizer.model_max_length  # == 1024 for Bart
            chunk_end += tokenizer.model_max_length  # == 1024 for Bart

                    # generate a summary on each batch
        summary_ids_lst = [model.generate(inputs, num_beams=4, max_length=100, early_stopping=True) for inputs in inputs_batch_lst]

                    # decode the output and join into one string with one paragraph per summary batch
        summary_batch_lst = []
        for summary_id in summary_ids_lst:
            summary_batch = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            summary_batch_lst.append(summary_batch[0])
        summary_all = '\n'.join(summary_batch_lst)
        st.success(summary_all)
        z =summary_all.split(" ")
        st.markdown(f'<h1 style="font-size:16px;">Number of Words in Output by BART: {+len(z)}</h1>', unsafe_allow_html=True)

else:
	st.warning("you need to provide an input URL")

