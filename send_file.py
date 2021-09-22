import requests
import httpx
from fastapi import UploadFile

#files = {'upload_file': open("dog_sound_test.wav", 'rb')}
#values = {'DB': 'photcat', 'OUT': 'wav', 'SHORT': 'short'}

#r = requests.post(url, files=files, data=values)
#test = fin.read()
#fin.close()
#files={"audio_file" : fin}
file =  open("dog_sound_test.wav", 'rb')
file.close()
#old_stringified = str(file.read())
#stringified = "LA" + old_stringified + "LA"
#with open("file.txt", "w") as f:
#    f.write(stringified)
file = open("file.txt", "r")
files = {"file": ("dog_sound", file, "multipart/form-data")}

r =  httpx.post(f"https://127.0.0.1:8000/upload_new_file/", files=files)
response = r.response()

print(response.status_code)