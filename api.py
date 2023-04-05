import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from summary import *

app = FastAPI()

logging.basicConfig(level=logging.DEBUG)

class SummarizeInput(BaseModel):
    url: str
    language: str

@app.post("/summarize")
def summarize(input_data: SummarizeInput):
    try:
        url = input_data.url
        language = input_data.language

        title, summarized_text = summarize_article(url, language)

        return {"title": title, "summarized_text": summarized_text}

    except Exception as e:
        logging.exception("Error while summarizing the article:")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
