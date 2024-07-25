from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_language(input: TextInput):
    text = input.text

    # Transform the input text using the vectorizer
    text_transformed = vectorizer.transform([text])

    # Predict the language using the model
    prediction = model.predict(text_transformed)
    
    # Get the language from the prediction
    predicted_language = prediction[0]

    return {"predicted_language": predicted_language}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
