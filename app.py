import streamlit as st
import google.generativeai as genai
import search_engine
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

st.set_page_config(page_title="SHL Recommender", page_icon="🎯", layout="centered")

@st.cache_resource
def load_recommender():
    return search_engine.AssessmentRecommender()
try:
    recommender = load_recommender()
except Exception as e:
    st.error(f"Error loading database: {e}")
    st.stop()
st.sidebar.title("⚙️ Settings")
num_results = st.sidebar.slider(
    "Number of Recommendations", 
    min_value=1, 
    max_value=10, 
    value=5,
    help="Slide to choose how many assessments to see."
)
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"Error configuring API: {e}")
    st.stop()
def get_recommendations(user_query, limit=5):
    prompt = f"""
    Analyze this Job Description and extract search keywords.
    JOB: "{user_query[:2000]}"
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
        tech_res = recommender.search(tech_q, k=15)
        beh_res = recommender.search(beh_q, k=15)
        results = []
        seen = set()
        while (tech_res or beh_res) and len(results) < limit:
            if tech_res:
                t = tech_res.pop(0)
                if t['url'] not in seen: results.append(t); seen.add(t['url'])
            if len(results) >= limit: break
            if beh_res:
                b = beh_res.pop(0)
                if b['url'] not in seen: results.append(b); seen.add(b['url'])
        return results
    except Exception as e:
        st.error(f"AI Error: {e}")
        return []
st.title("🎯 SHL Assessment Recommender")
st.markdown("Enter a Job Description to get AI-powered assessment suggestions.")
with st.form("query_form"):
    user_query = st.text_area(
        "Job Description:", 
        height=150,
        placeholder="Example: Need a Senior Java Developer with good leadership skills..."
    )
    submitted = st.form_submit_button("Get Recommendations 🚀")
if submitted and user_query:
    with st.spinner("Analyzing..."):
        results = get_recommendations(user_query, limit=num_results)
        if results:
            st.success(f"Found {len(results)} matches!")
            for item in results:
                with st.expander(f"📄 {item['name']}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Description:** {item['description'][:300]}...")
                        st.markdown(f"**Type:** {', '.join(item['test_type'])}")
                    with col2:
                        st.markdown(f"⏱ **{item['duration']} min**")
                        if item['remote_support'] == "Yes": st.caption("✅ Remote")
                    st.markdown(f"🔗 [View Assessment]({item['url']})")
        else:
            st.warning("No results found.")