from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from typing import List
from pydantic import BaseModel

import torchaudio

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import uvicorn

model = MusicGen.get_pretrained('small', device='cpu')

app = FastAPI()

class Item(BaseModel):
    descriptions: List[str]
    duration: int

@app.post("/generate/")
async def generate_audio(item: Item):
    model.set_generation_params(duration=item.duration) 
    wav = model.generate(item.descriptions) 
    filename = ''
    for idx, one_wav in enumerate(wav):
        filename = f'{idx}.wav'
        audio_write(filename, one_wav.cpu(), model.sample_rate, strategy="loudness")
    return {'filename': filename} 

@app.get("/download/{filename}")
async def download_audio(filename: str):
    return FileResponse(filename, media_type="audio/wav", filename=filename)  


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)