from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
import os
import uvicorn
import numpy as np

from config import Config
from typing import Literal
from train_bis import AudioNet

import torch, torchaudio
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import librosa

from train_bis import ESC50Dataset, AudioNet

# ???
datapath = Path('/home/samed/Documents/codementor/tim_ob/')
datapath.exists()

csv = pd.read_csv(datapath / Path('ESC-50-master/meta/esc50.csv'))

torchaudio.set_audio_backend("sox_io")
x, sr = torchaudio.load(datapath / "ESC-50-master/audio" / "4-165845-A-45.wav")

# Building data loaders
train_data = ESC50Dataset(folds=[1,2,3])
val_data = ESC50Dataset(folds=[4])
test_data = ESC50Dataset(folds=[5])

# Loading data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=8)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8)

pl.seed_everything(0)

# Test that the network works on a single mini-batch
audionet = AudioNet()
xb, yb = next(iter(train_loader))
print(audionet(xb).shape)

trainer = pl.Trainer(gpus=0, max_epochs=20)
trainer.fit(audionet, train_loader, val_loader)
trainer.test(audionet, test_loader)

# Audio file loading parameters
sample_rate = 8000

resample = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=sample_rate
        )

melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate)

db = torchaudio.transforms.AmplitudeToDB(top_db=80)


# Function to load file
def load_wav_file(wav_file : str) -> torch.Tensor:
    wav, _ = torchaudio.load(Config.repository + "/" + wav_file)
    xb = db(
            melspec(
            resample(wav)
        )
    )
    return xb

# Loading the correspondance between labels and targets
meta = pd.read_csv("ESC-50-master/meta/esc50.csv")
label2target = meta[["target", "category"]].groupby("target").agg("first").category.to_dict()

# Function to perform prediction
def predict_outcome(xb : torch.Tensor) -> str:
    softmax_scores = audionet(xb.unsqueeze(1))
    prediction = np.argmax(softmax_scores.detach().numpy()[0])
    return label2target[prediction]


app = FastAPI()

@app.get("/get_sound_class_for_record")
async def get_sound_class(file_name : str) -> str:
    # Asserting that we got a proper string for a file that actually exists 
    # in our repository
    try:
        assert(file_name in os.listdir(Config.repository))
    except AssertionError:
        return "File was not found in our repository. Please add the file or use an existing file."

    # Loading file
    xb = load_wav_file(file_name)

    # Model prediction
    prediction = predict_outcome(xb).replace("_", " ")

    # Response retrieval
    return f"Sound class : {prediction}"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


def is_valid_ip_address(address):
    if not isinstance(address, str):
        return False
    parts = address.split(".")
    if len(parts) != 4:
        return False
    if any(not part.isnumeric() for part in parts):
        return False
    if any(int(part) > 255 for part in parts):
        return False
    return True
