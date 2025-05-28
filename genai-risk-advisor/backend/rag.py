import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from fastapi import HTTPException

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing or invalid.")

def analyze_risk(applicant):
    llm = ChatOpenAI(
        openai_api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model="mistral-saba-24b",  # Updated model name here
        temperature=0
    )

    pdf_path = "app/sample_docs/all_claims.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    prompt = f"""
Applicant Profile:
- Name: {applicant['name'].strip()}
- Age: {applicant['age']}
- Location: {applicant['location'].strip()}
- Occupation: {applicant['occupation'].strip()}
- Claim Description: {applicant['claim_description'].strip()}

Based on the internal knowledge (claims.pdf), rate the applicant’s insurance risk as Low, Medium, or High. 
Provide 2–3 reasons referencing similar past claims.

Important: Only compare to other locations within the United Kingdom. Do not reference locations outside the UK.
"""

    try:
        result = qa.run(prompt)
        return {"risk_level": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
