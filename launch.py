import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import openai


openai.api_key = os.environ.get("OPENAI_API_KEY")


# Load pre-defined vector databse
db = FAISS.load_local("./offers_faiss_index", OpenAIEmbeddings())


st.title("🏷️ Search for the best offers")
query = st.text_input(
    "Start your search 👇",
    placeholder="Try shoes..."
)

st.button("Search")




if query is not None:


    progress_bar = st.progress(0, "Fetching the best offers for you..")
    
    docs = db.similarity_search_with_relevance_scores(query)
    
    progress_bar.progress(1.0, "Offers found:")



    for doc, score in docs:
        st.markdown(f"* Offer: {doc.page_content}, Score: {score}")

        




