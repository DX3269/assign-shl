import uvicorn
import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
import search_engine

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

recommender = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    try:
        recommender = search_engine.AssessmentRecommender()
    except Exception as e:
        print(f"Lifespan Error: {e}")
    yield

app = FastAPI(title="SHL Assessment Recommender API", lifespan=lifespan)

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception:
    model = None

class QueryRequest(BaseModel):
    query: str
    limit: int = 5

class AssessmentResponse(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendResponse(BaseModel):
    recommended_assessments: List[AssessmentResponse]

def extract_search_terms(user_query):
    if not model: return user_query, "Personality"
    
    prompt = f"""
    Analyze Job Description: "{user_query[:2000]}"
    Output strictly:
    TECHNICAL: <keywords>
    BEHAVIORAL: <keywords>
    """
    try:
        response = model.generate_content(prompt)
        text = response.text
        tech_q, beh_q = text, "Personality"
        for line in text.split('\n'):
            if "TECHNICAL:" in line: tech_q = line.split(":")[1].strip()
            elif "BEHAVIORAL:" in line: beh_q = line.split(":")[1].strip()
        return tech_q, beh_q
    except:
        return "Skills", "Personality"

def balance_results(tech_results, beh_results, total_needed=10):
    final_mix = []
    while len(final_mix) < total_needed:
        if not tech_results and not beh_results: break
        if tech_results:
            t = tech_results.pop(0)
            if t['url'] not in [x['url'] for x in final_mix]: final_mix.append(t)
        if len(final_mix) >= total_needed: break
        if beh_results:
            b = beh_results.pop(0)
            if b['url'] not in [x['url'] for x in final_mix]: final_mix.append(b)
    return final_mix

@app.get("/health")
def health_check():
    return {"status": "healthy", "engine_loaded": recommender is not None}

@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(request: QueryRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Server is starting up. Try again in 30 seconds.")

    try:
        safe_limit = max(1, min(request.limit, 10))
        tech_q, beh_q = extract_search_terms(request.query)
        
        tech_res = recommender.search(tech_q, k=20)
        beh_res = recommender.search(beh_q, k=20)
        
        balanced = balance_results(tech_res, beh_res, total_needed=safe_limit)
        
        formatted = []
        for item in balanced:
            formatted.append({
                "url": item['url'],
                "name": item['name'],
                "adaptive_support": item['adaptive_support'],
                "description": item['description'],
                "duration": item['duration'],
                "remote_support": item['remote_support'],
                "test_type": item['test_type']
            })
        return {"recommended_assessments": formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)