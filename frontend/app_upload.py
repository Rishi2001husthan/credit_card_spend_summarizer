import streamlit as st
import requests

st.set_page_config(
    page_title="Document Ingestion Engine",
    page_icon="📥",
    layout="wide"
)

st.title("📥 Multimodal Ingestion Pipeline Dashboard")
st.subheader("Process credit rules, guidelines, or charts into your vector space.")
st.markdown("---")

uploaded_files = st.file_uploader(
    "Select operational documents (PDF files supported):",
    type=["pdf"],
    accept_multiple_files=True
)

target_collection = st.text_input("Active Target Schema Collection Identifier:", value="KB_Credit_Card_Spend_Summarizer.pdf")

if st.button("🚀 Process and Tokenize Uploaded Materials"):
    if not uploaded_files:
        st.warning("Please upload at least one valid PDF before processing.")
    else:
        # FastAPI backend streaming url target endpoint
        FASTAPI_URL = "http://127.0.0.1:8000/api/v1/ingest"

        for file_block in uploaded_files:
            with st.status(f"Uploading file: {file_block.name} to ingestion microservice...", expanded=True) as status:
                
                # Read binary payload from Streamlit cache memory
                file_bytes = file_block.read()
                
                # Build multi-part form structures cleanly
                payload_files = {
                    "file": (file_block.name, file_bytes, "application/pdf")
                }
                form_data = {
                    "collection_name": target_collection
                }
                
                try:
                    # POST request straight down to the FastAPI backend port
                    response = requests.post(FASTAPI_URL, files=payload_files, data=form_data)
                    
                    if response.status_code == 200:
                        status.update(label=f"Successfully processed {file_block.name}!", state="complete")
                        st.success(f"Backend confirmation details: {response.json().get('status')}")
                    else:
                        status.update(label=f"❌ Processing Error on {file_block.name}", state="error")
                        st.error(f"HTTP Status {response.status_code}: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    status.update(label="❌ Connection Timeout", state="error")
                    st.error("Could not bridge connectivity to FastAPI backend server. Ensure it is actively running on Port 8000!")