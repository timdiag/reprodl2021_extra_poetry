from fastapi import FastAPI, File, UploadFile
import os
import uvicorn
from typing import Optional

app = FastAPI()


@app.post("/upload_new_file/")
async def upload_new_file(upload_new_file : UploadFile = File(...)) -> str:
    file_name = "new_file"
    try:
        contents = eval(upload_new_file.file.read()[2:-2])
        with open(f"ESC-50-master/audio/{file_name}.wav", "wb") as f:
            f.write(contents)
        return f"File {file_name}.wav has been stored successfully."
    except:
        return "An error occured."