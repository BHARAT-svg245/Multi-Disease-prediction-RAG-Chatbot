import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI

from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field
from typing import Optional

from google import genai

from prompt_module import prompt


tab1,tab2=st.tabs(["Disease Prediction","RAG Chatbot"])

with tab1:
    # ---------------------------------
    # Page Configuration
    # ---------------------------------
    st.set_page_config(
        page_title="Multi Disease Classification",
        page_icon="🩺",
        layout="centered"
    )
    
    st.title("🩺 Multi Disease Classification")
    st.write(
        "Upload a medical image. The application will first identify the organ "
        "and then predict the corresponding disease."
    )
    
    # ---------------------------------
    # Load Models
    # ---------------------------------
    @st.cache_resource
    def load_models():
        return {
            "organ": tf.keras.models.load_model("Model/Organ_classifier_256.keras"),
            "Brain": tf.keras.models.load_model("Model/Brain_classifier_256.keras"),
            "Lung": tf.keras.models.load_model("Model/Lung_classifier_256.keras"),
            "Breast": tf.keras.models.load_model("Model/Breast_Cancer_classifier_256.keras"),
            "Eye": tf.keras.models.load_model("Model/Eye_classifier_256.keras"),
            "Skin": tf.keras.models.load_model("Model/Skin_classifier_256.keras"),
        }
    
    models = load_models()
    
    # ---------------------------------
    # Classes
    # ---------------------------------
    organ_classes = [
        "Brain",
        "Breast",
        "Lung",
        "Eye",
        "Skin"
    ]
    
    disease_classes = {
        "Brain": [
            "Glioma",
            "Meningioma",
            "No Tumor",
            "Pituitary"
        ],
    
        "Breast": [
            "Benign",
            "Malignant"
        ],
    
        "Skin": [
            "Benign",
            "Malignant"
        ],
    
        "Lung": [
            "Colon Adenocarcinoma",
            "Colon Benign",
            "Lung Adenocarcinoma",
            "Lung Benign",
            "Lung Squamous Cell Carcinoma"
        ],
    
        "Eye": [
            "Age Related Degeneration",
            "Cataract",
            "Diabetic Retinopathy",
            "Glaucoma",
            "Hypertensive Retinopathy",
            "Myopia",
            "Normal"
        ]
    }
    
    IMG_SIZE = (256, 256)
    
    # ---------------------------------
    # Image Preprocessing
    # ---------------------------------
    def preprocess_image(image):
    
        image = image.convert("RGB")
        image = image.resize(IMG_SIZE)
    
        img = np.array(image).astype("float32")
    
        img = np.expand_dims(img, axis=0)
    
        return img
    
    # ---------------------------------
    # Upload Image
    # ---------------------------------
    uploaded_files = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
            st.success(f"{len(uploaded_files)} image(s) uploaded successfully.")
        
            # Three cards in one row
            cols = st.columns(3)
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                display_img=image.resize(IMG_SIZE)
                img = preprocess_image(image)
        
                # ---------------- Organ Prediction ----------------
                organ_prediction = models["organ"].predict(img, verbose=0)
                organ_index = np.argmax(organ_prediction)
                predicted_organ = organ_classes[organ_index]
                organ_confidence = (organ_prediction[0][organ_index] * 100)
                # ---------------- Disease Prediction ----------------
                disease_model = models[predicted_organ]  
                disease_prediction = disease_model.predict(
                    img,
                    verbose=0
                )
                # ------------------- Viewing Image -------------------
                st.image(display_img,use_container_width=True)

                disease_index = np.argmax(disease_prediction)
                predicted_disease = disease_classes[predicted_organ][disease_index]
                disease_confidence = ( disease_prediction[0][disease_index] * 100)
                # Main Organ Details
                st.metric("Detected Organ",predicted_organ)             
                st.write(f"Confidence : **{organ_confidence:.2f}%**")

                # Predicted Disease type in Organ
                st.metric("Detected Disease",predicted_disease)
                #st.write(f"Confidence: **{organ_confidence:.2f}%**")
                st.success("Prediction Completed")
                
                            
                st.progress(float(disease_confidence / 100))

with tab2:
    st.set_page_config(
        page_title="Multi-Disease Assistant",
        page_icon="🩺",
        layout="centered"
    )
    
    st.title("🩺 Multi-Disease Assistant")
    st.write(
        "Ask General questions related to disease like symptoms, diagnosis, preventions, diets, Medication and Treatment and Management"
    )
    
    @st.cache_resource
    def load_embeddings():
        return HuggingFaceEmbeddings(model_name="Place Your model")
    embeddings = load_embeddings()
    
    @st.cache_resource
    def load_vectorstore(_embeddings):
        return FAISS.load_local("Multi_Disease",_embeddings,allow_dangerous_deserialization=True)
    db = load_vectorstore(embeddings)
    
    @st.cache_resource
    def load_documents(_db):
        return list(_db.docstore._dict.values())
    documents = load_documents(db)
    
    # Getting the title for metadata filtering automation.
    available_titles = sorted(
        {
            doc.metadata.get("title")
            for doc in documents
            if doc.metadata.get("title")
        }
    )
    
    @st.cache_resource
    def load_bm25(_doc):
        retriever = BM25Retriever.from_documents(_doc)
        retriever.k = 5
        return retriever
    bm25_retriever = load_bm25(documents)
    
    class QueryRoute(BaseModel):
        query: str = Field(description="The semantic search query used for vector retrieval.")
        title: Optional[str] = Field(
            default=None,
            description="Specific disease title to filter by, or null if no specific disease is mentioned")
    
    @st.cache_resource
    def load_llm():
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0,api_key=" Your API Key")
    llm = load_llm()
    
    router = llm.with_structured_output(QueryRoute)
    
    @st.cache_resource
    def load_reranker():
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = load_reranker()
    
    routing_prompt = prompt()
    
    query_router = routing_prompt | router
    
    @st.cache_resource
    def load_google_client():
        return genai.Client(api_key="Your API key")
    client = load_google_client()
    
    query = st.text_input(
        "Enter your question",
        placeholder="e.g. What are the symptoms of diabetes?"
    )
    
    if st.button("Ask", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Understanding your question..."):
                    route = query_router.invoke(
                        {
                            "query": query,
                            "available_titles": ", ".join(
                                available_titles
                            )
                        }
                    )
                with st.spinner("Searching documents..."):
                    if route.title is None:
                        dense_docs = db.similarity_search(query,k=15)
                    else:
                        dense_docs = db.similarity_search(query,k=15,filter={"title": route.title})
    
                with st.spinner("Performing keyword retrieval..."):
                    sparse_docs = bm25_retriever.invoke(
                        route.query
                    )
    
                total_docs = dense_docs + sparse_docs
                unique_docs = {}
                for doc in total_docs:
                    if doc.page_content not in unique_docs:
                        unique_docs[doc.page_content] = doc
    
                with st.spinner("Reranking relevant documents..."):
    
                    pairs = [[route.query,doc.page_content] for doc in unique_docs.values()]
                    scores = reranker.predict(pairs)
                    ranked_docs = sorted(zip(unique_docs.values(),scores), 
                    key=lambda x: x[1],reverse=True)
                    top_docs = [doc for doc, score in ranked_docs[:5]]
    
                context = "\n\n".join([doc.page_content for doc in top_docs])
    
                final_prompt = f"""
    You are a Multi-Disease Assistant.
    
    Use ONLY the information provided in the context below
    to answer the user's question.
    
    If the answer is not present in the context, reply exactly:
    
    "I could not find this information in the provided documents."
    
    Do not use outside knowledge.
    
    Context:
    {context}
    Question:
    {query}
    Answer:
    """
                with st.spinner("Generating answer..."):
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=final_prompt
                    )
                    answer = response.text
                st.subheader("Answer")
                st.write(answer)
    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    
       
                