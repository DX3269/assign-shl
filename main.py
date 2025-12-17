import uvicorn
import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import search_engine  


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 


try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    print("Gemini AI successfully configured.")
except Exception as e:
    print(f"Warning: Gemini configuration failed. LLM features will not work. Error: {e}")


app = FastAPI()
print("Initializing Search Engine...")

recommender = search_engine.AssessmentRecommender()

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
   
    prompt = f"""
    You are an expert HR Recruiter. Extract search keywords from this job query.
    Query: "{user_query}"
    Task:
    1. Identify 'Technical Skills' (Hard skills, tools, languages).
    2. Identify 'Behavioral Skills' (Soft skills, personality, competencies).
    Output strictly in this format:
    TECHNICAL: <comma separated keywords>
    BEHAVIORAL: <comma separated keywords>
    """
    try:
        response = model.generate_content(prompt)
        text = response.text
        tech_query = user_query 
        beh_query = "Personality behavior collaboration" 
        for line in text.split('\n'):
            if "TECHNICAL:" in line:
                cleaned = line.split("TECHNICAL:")[1].strip()
                if cleaned: tech_query = cleaned
            elif "BEHAVIORAL:" in line:
                cleaned = line.split("BEHAVIORAL:")[1].strip()
                if cleaned: beh_query = cleaned
        return tech_query, beh_query

    except Exception as e:
        print(f"LLM Error (Using fallback): {e}")
        return user_query, "Personality behavior"


def balance_results(tech_results, beh_results, total_needed=10):
    
    final_mix = []
    while len(final_mix) < total_needed:
        if not tech_results and not beh_results:
            break 
        if tech_results:
            item = tech_results.pop(0)
            if item['url'] not in [x['url'] for x in final_mix]:
                final_mix.append(item)
        if len(final_mix) >= total_needed: break
        if beh_results:
            item = beh_results.pop(0)
            if item['url'] not in [x['url'] for x in final_mix]:
                final_mix.append(item)
    return final_mix

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(request: QueryRequest):
    try:
        print(f"\nNew Request: {request.query[:50]}...")
        safe_limit = max(1, min(request.limit, 10))
        tech_keywords, beh_keywords = extract_search_terms(request.query)
        print(f" -> LLM Technical Terms: [{tech_keywords}]")
        print(f" -> LLM Behavioral Terms: [{beh_keywords}]")
        tech_results = recommender.search(tech_keywords, k=10)
        beh_results = recommender.search(beh_keywords, k=10)
        balanced = balance_results(tech_res, beh_res, total_needed=safe_limit)
        formatted_list = []
        for item in balanced:
            formatted_list.append({
                "url": item['url'],
                "name": item['name'],
                "adaptive_support": item['adaptive_support'],
                "description": item['description'],
                "duration": item['duration'],
                "remote_support": item['remote_support'],
                "test_type": item['test_type']
            })
        return {"recommended_assessments": formatted_list}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

        
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)