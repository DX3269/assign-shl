import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


current_script_dir = os.path.dirname(os.path.abspath(__file__))

VECTOR_DB_FOLDER = os.path.join(current_script_dir, "faiss_index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
print(f"Looking for database at: {VECTOR_DB_FOLDER}")

class AssessmentRecommender:
    def __init__(self):
        if not os.path.exists(VECTOR_DB_FOLDER):
            raise FileNotFoundError(f"CRITICAL ERROR: Could not find '{VECTOR_DB_FOLDER}'.\n"
                                    "Make sure the 'faiss_index' folder is inside the 'Assign' folder!")
        print("Loading Vector Database...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_db = FAISS.load_local(
            VECTOR_DB_FOLDER, 
            self.embeddings, 
            allow_dangerous_deserialization=True 
        )
        print("Database Loaded Successfully.")

    def search(self, query, k=10):
        results = self.vector_db.similarity_search_with_score(query, k=k*2)
        recommendations = []
        seen_urls = set()
        for doc, score in results:
            meta = doc.metadata
            url = meta.get("url")
            if url in seen_urls: continue
            seen_urls.add(url)
            recommendations.append({
                "name": meta.get("name"),
                "url": url,
                "description": meta.get("description"),
                "duration": meta.get("duration"),
                "test_type": meta.get("test_type"), 
                "adaptive_support": meta.get("adaptive_support"),
                "remote_support": meta.get("remote_support"),
                "score": float(score)
            })
            if len(recommendations) >= k:
                break
        return recommendations
        
if __name__ == "__main__":
    try:
        recommender = AssessmentRecommender()
        print("Test Success: Database loaded.")
    except Exception as e:
        print(e)