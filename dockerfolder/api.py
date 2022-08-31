
from fastapi import FastAPI, File
import uvicorn
import torch
import pickle
from data_processor import ImageProcessor, TextProcessor
from fastapi import UploadFile
from fastapi import Form
from fastapi.responses import JSONResponse
from PIL import Image
from neural_network_models import CombinedModel

text_processor = TextProcessor()
image_processor = ImageProcessor()
with open('models/decoder.pickle', 'rb') as f:
    decoder = pickle.load(f)

model = CombinedModel(decoder=decoder)
model.load_state_dict(torch.load("models/combined_model_state.pt", map_location=torch.device("cpu")))

app = FastAPI()

@app.post('/predictor')
def combined_predictor(image : UploadFile = File(...), text: str = Form(...)):
    processed_image = image_processor(Image.open(image.file))
    processed_text = text_processor(text)
    prediction = model.predict(processed_image, processed_text)
    probs = model.predict_proba(processed_image, processed_text)
    classes = model.predict_classes(processed_image, processed_text)
    print(prediction)
    print(probs)
    print(classes)
    return JSONResponse(status_code=200, content={'prediction': prediction.tolist(), 'probs': probs.tolist(), 'classes': classes})

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port="8080")
 

