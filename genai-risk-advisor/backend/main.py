from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import analyze_risk

app = FastAPI()

# Allow CORS for frontend running on localhost:3000
origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected input format
class Applicant(BaseModel):
    name: str
    age: int
    location: str
    occupation: str
    claim_description: str

# Endpoint to receive applicant data and run analysis
@app.post("/analyze-risk")
def analyze(applicant: Applicant):
    return analyze_risk(applicant.dict())
