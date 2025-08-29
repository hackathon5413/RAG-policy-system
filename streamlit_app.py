import os
import uuid
from datetime import datetime

import boto3
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# S3 Configuration
S3_BUCKET = os.getenv('S3_BUCKET', 'personal-v1-common')
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-south-1')
)

st.set_page_config(
    page_title="HackRX Document Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 HackRX Document Q&A")

# Initialize session state
if 'questions' not in st.session_state:
    st.session_state.questions = [""]
if 'input_method' not in st.session_state:
    st.session_state.input_method = "upload"  # Default to file upload

def add_question():
    st.session_state.questions.append("")

def remove_question(index):
    if len(st.session_state.questions) > 1:
        st.session_state.questions.pop(index)

def switch_input_method():
    st.session_state.input_method = "url" if st.session_state.input_method == "upload" else "upload"

# API Configuration
API_ENDPOINT = "http://localhost:8080/api/v1/hackrx/run"  # Update with your actual endpoint

# Function to upload file to S3
def upload_to_s3(file):
    try:
        # Generate a unique filename
        file_extension = file.name.split('.')[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"

        # Upload to S3
        try:
            s3_client.upload_fileobj(
                file,
                S3_BUCKET,
                unique_filename,
                ExtraArgs={'ContentType': 'application/octet-stream'}
            )
        except s3_client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'AccessDenied':
                st.error("Access denied. Please check your AWS credentials.")
            else:
                st.error(f"S3 upload error: {e!s}")
            return None
        except Exception as e:
            st.error(f"Unexpected error during upload: {e!s}")
            return None

        # Generate the presigned URL that will work for 1 hour
        s3_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': unique_filename},
            ExpiresIn=3600  # URL valid for 1 hour
        )
        return s3_url
    except Exception as e:
        st.error(f"Error preparing file: {e!s}")
        return None

# Document input section
st.subheader("Document Input")
with st.expander("Change input method", expanded=False):
    st.button(
        "Switch to " + ("URL" if st.session_state.input_method == "upload" else "File Upload"),
        on_click=switch_input_method
    )

document_url = None

if st.session_state.input_method == "upload":
    # File upload is now the default
    uploaded_file = st.file_uploader(
        "Upload your document",
        type=['pdf', 'txt', 'doc', 'docx', 'xlsx', 'xls', 'pptx', 'ppt', 'png', 'jpg', 'jpeg', 'zip'],
        help="Supported formats: PDF, Word, Excel, PowerPoint, Images, and ZIP files"
    )
    if uploaded_file:
        with st.spinner("Uploading document..."):
            document_url = upload_to_s3(uploaded_file)
            if document_url:
                st.success("Document uploaded successfully! 🎉")
else:
    # URL input is hidden by default
    document_url = st.text_input("Document URL", placeholder="Enter the document URL")

st.subheader("Questions")

# Dynamic question inputs
for i, question in enumerate(st.session_state.questions):
    col1, col2 = st.columns([6, 1])
    with col1:
        st.session_state.questions[i] = st.text_input(
            f"Question {i+1}",
            value=question,
            key=f"q_{i}",
            placeholder="Enter your question"
        )
    with col2:
        if i > 0 and st.button("❌", key=f"remove_{i}"):  # Don't allow removing the first question
            remove_question(i)
            st.rerun()

# Add question button
if st.button("+ Add Another Question"):
    add_question()
    st.rerun()

# Submit button
submitted = st.button("Submit")

if submitted:
    if not document_url:
        st.error("Please enter a document URL")
    elif not any(q.strip() for q in st.session_state.questions):
        st.error("Please enter at least one question")
    else:
        # Show loading spinner
        with st.spinner("Processing your request..."):
            try:
                # Prepare the request
                headers = {
                    "Content-Type": "application/json"
                }

                payload = {
                    "documents": document_url,
                    "questions": [q for q in st.session_state.questions if q.strip()]
                }

                # Make the API call
                response = requests.post(API_ENDPOINT,
                                      headers=headers,
                                      json=payload)

                if response.status_code == 200:
                    data = response.json()

                    # Display results
                    st.success("✅ Successfully processed the document!")

                    st.subheader("Answers")
                    for i, (question, answer) in enumerate(zip(payload["questions"], data["answers"], strict=True)):
                        with st.expander(f"Q{i+1}: {question}", expanded=True):
                            st.markdown(answer)
                            st.button(
                                "Copy",
                                key=f"copy_{i}",
                                on_click=lambda a=answer: st.write(a)
                            )
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"An error occurred: {e!s}")

# Add some helpful instructions in the sidebar
with st.sidebar:
    st.markdown("""
    ### Welcome to Document Q&A! 🎉

    This is a free tool that helps you get answers from documents.

    ### How to use:
    1. Choose your input method:
       - Paste a document URL, or
       - Upload a local file (PDF, DOC, TXT)
    2. Add your question(s)
    3. Click Submit to get answers

    ### Tips:
    - Supported file types: PDF, DOC, DOCX, TXT
    - Add multiple questions for better insights
    - Use ❌ to remove unwanted questions
    - Expand/collapse answers as needed
    - Copy answers with one click

    ### About
    This tool uses advanced AI to analyze documents
    and provide relevant answers to your questions.
    Your uploaded files are securely stored and
    accessible for analysis.
    """)
