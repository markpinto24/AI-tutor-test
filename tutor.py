import os
import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
import ffmpeg
import whisper
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Any
import mimetypes

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add these imports at the top
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI GPT Model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# Add this after initializing OpenAI
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Define Preloaded Documents Directory
PRELOADED_DOCS_FOLDER = "files"


# Define State (Shared Memory Between Interactions)
class TutorState(TypedDict):
    extracted_content: str
    chat_history: List[str]
    vectorstore: Any


# Global state variable; in a real app, use per-user sessions
tutor_state: TutorState = {
    "extracted_content": "", 
    "chat_history": [],
    "vectorstore": None
}

# ---------------- Processing Functions ----------------


def extract_text_from_pdfs():
    all_text = ""
    if not os.path.exists(PRELOADED_DOCS_FOLDER):
        os.makedirs(PRELOADED_DOCS_FOLDER)
    pdf_files = [f for f in os.listdir(PRELOADED_DOCS_FOLDER) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PRELOADED_DOCS_FOLDER, pdf_file)
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join(page.get_text("text") for page in doc)
            all_text += f"\n\n--- Document: {pdf_file} ---\n{text}"
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {str(e)}")
    return all_text.strip()


def extract_text_from_images():
    all_text = ""
    image_files = [
        f
        for f in os.listdir(PRELOADED_DOCS_FOLDER)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    for image_file in image_files:
        image_path = os.path.join(PRELOADED_DOCS_FOLDER, image_file)
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            _, threshold = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
            text = pytesseract.image_to_string(threshold)
            all_text += f"\n\n--- Image: {image_file} ---\n{text.strip()}"
        except Exception as e:
            print(f"❌ Error processing {image_file}: {str(e)}")
    return all_text.strip()


def transcribe_videos():
    all_text = ""
    video_files = [
        f
        for f in os.listdir(PRELOADED_DOCS_FOLDER)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]
    for video_file in video_files:
        video_path = os.path.join(PRELOADED_DOCS_FOLDER, video_file)
        audio_path = "./temp_audio.wav"
        try:
            ffmpeg.input(video_path).output(
                audio_path, format="wav", acodec="pcm_s16le", ac=1, ar="16k"
            ).run(overwrite_output=True)
            model = whisper.load_model("base", device="cpu")
            result = model.transcribe(audio_path, fp16=False)
            all_text += f"\n\n--- Video: {video_file} ---\n{result['text']}"
        except Exception as e:
            print(f"❌ Error processing {video_file}: {str(e)}")
    return all_text.strip()


def extract_text_from_txt():
    all_text = ""
    text_files = [f for f in os.listdir(PRELOADED_DOCS_FOLDER) if f.endswith(".txt")]
    for text_file in text_files:
        text_path = os.path.join(PRELOADED_DOCS_FOLDER, text_file)
        try:
            with open(text_path, "r", encoding="utf-8") as file:
                content = file.read()
                all_text += f"\n\n--- Document: {text_file} ---\n{content.strip()}"
        except Exception as e:
            print(f"❌ Error processing {text_file}: {str(e)}")
    return all_text.strip()


def create_vectorstore(text: str) -> FAISS:
    """
    Creates a FAISS vector store from the input text
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Split text into chunks
    texts = text_splitter.split_text(text)

    # # Print the number of chunks
    # print(f"Number of chunks: {len(texts)}")

    # Print each chunk individually
    for i, chunk in enumerate(texts):
        print(f"Chunk {i + 1}:")
        print(chunk)
        print("-" * 40)  # Separator line for easier reading
    
    # Create documents
    documents = [Document(page_content=t) for t in texts]
    
    # Create and return the vector store
    return FAISS.from_documents(documents, embeddings)

# ---------------- AI Tutor Function ----------------


def answer_question(state: TutorState, user_question: str) -> str:
    """
    Uses GPT to answer questions based on vectorized content.
    """
    state["chat_history"] = state["chat_history"][-5:]
    
    if state["vectorstore"] is None:
        return "No study materials have been loaded and vectorized."
    
    # Search for relevant documents
    relevant_docs = state["vectorstore"].similarity_search(
        user_question,
        k=3  # Get top 3 most relevant chunks
    )
    
    # Combine relevant texts
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    chat_context = " \n ".join(state["chat_history"])
    
    prompt = f"""
You are an AI Tutor for school students. You must **only** answer based on the provided study materials.

- If the question is related to the content, provide a detailed, educational answer.
- If the question is not found in the study materials, respond with:
  "I'm sorry, but I can only answer questions based on the provided study material. Please ask something related to the content."
**Study Material:**
- If the question is not found in the study materials, but is related or connected to the content, you can still answer it.
{context}

**Previous Conversation:**
{chat_context}

**Student's Question:**
{user_question}
"""
    
    response = llm.invoke(prompt)
    state["chat_history"].append(f"User: {user_question}\nAI: {response.content}")
    return response.content



# ---------------- FastAPI Application ----------------

app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request and response
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    chat_history: List[str]


@app.on_event("startup")
def load_study_materials():
    """
    On startup, load study materials from PDFs, images, videos, and text files.
    """
    global tutor_state
    print("🔹 Loading study materials...")
    extracted_text = "\n".join(
        [
            extract_text_from_pdfs(),
            extract_text_from_images(),
            transcribe_videos(),
            extract_text_from_txt(),
        ]
    ).strip()
    
    if not extracted_text:
        print("❌ No study materials found in folder 'files'.")
    else:
        # Create vector store
        tutor_state["vectorstore"] = create_vectorstore(extracted_text)
        
    tutor_state["extracted_content"] = extracted_text
    tutor_state["chat_history"] = []
    print("✅ Study materials loaded and vectorized.")
    # print(extracted_text)


@app.get("/")
def read_root():
    return {"message": "API is running!"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question_endpoint(request: QuestionRequest):
    """
    Endpoint to ask a question.
    """
    if not tutor_state.get("extracted_content"):
        raise HTTPException(status_code=400, detail="No study materials available.")
    answer = answer_question(tutor_state, request.question)
    return AnswerResponse(answer=answer, chat_history=tutor_state["chat_history"])


@app.get("/chat_history", response_model=List[str])
def get_chat_history():
    """
    Endpoint to get the current chat history.
    """
    return tutor_state.get("chat_history", [])

@app.delete("/clear_history", response_model=List[str])
def clear_chat_history():
    """
    Endpoint to clear the chat history.
    """
    global tutor_state
    tutor_state["chat_history"] = []
    return tutor_state["chat_history"]