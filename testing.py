import os
import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
import ffmpeg
import whisper
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Any, Dict, Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add these imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise EnvironmentError("OPENAI_API_KEY not found. Please set it in your .env file.")

# Initialize OpenAI GPT Model with error handling
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

# Define Preloaded Documents Directory
PRELOADED_DOCS_FOLDER = "files"
os.makedirs(PRELOADED_DOCS_FOLDER, exist_ok=True)

# Define State (Shared Memory Between Interactions)
class TutorState(TypedDict):
    extracted_content: str
    chat_history: List[str]
    vectorstore: Optional[Any]


# Global state variable
tutor_state: TutorState = {
    "extracted_content": "", 
    "chat_history": [],
    "vectorstore": None
}

# ---------------- Processing Functions ----------------

async def process_files_async():
    """Process all files asynchronously."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        # Run file processing tasks concurrently
        tasks = [
            loop.run_in_executor(executor, extract_text_from_pdfs),
            loop.run_in_executor(executor, extract_text_from_images),
            loop.run_in_executor(executor, transcribe_videos),
            loop.run_in_executor(executor, extract_text_from_txt)
        ]
        results = await asyncio.gather(*tasks)
    
    # Combine results
    extracted_text = "\n".join([result for result in results if result]).strip()
    
    if extracted_text:
        # Create vector store
        tutor_state["extracted_content"] = extracted_text
        tutor_state["vectorstore"] = create_vectorstore(extracted_text)
        logger.info(f"Files processed and vectorized successfully.")
        logger.info(f"Extracted {len(extracted_text)} characters of content from documents.")
    else:
        logger.warning("No content extracted from files in the 'files' directory.")


def extract_text_from_pdfs() -> str:
    """Extract text from PDF files."""
    if not os.path.exists(PRELOADED_DOCS_FOLDER):
        logger.warning(f"Folder {PRELOADED_DOCS_FOLDER} does not exist")
        return ""
    
    all_text = ""
    pdf_files = [f for f in os.listdir(PRELOADED_DOCS_FOLDER) if f.lower().endswith(".pdf")]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PRELOADED_DOCS_FOLDER, pdf_file)
        try:
            doc = fitz.open(pdf_path)
            # Process each page with improved text extraction
            page_texts = []
            for page in doc:
                # Get text with better layout preservation
                text = page.get_text("text")
                page_texts.append(text)
            
            # Join all pages with proper separation
            text = "\n\n".join(page_texts)
            all_text += f"\n\n--- Document: {pdf_file} ---\n{text}"
            doc.close()  # Properly close the document to free resources
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_file}: {str(e)}")
    
    return all_text.strip()


def extract_text_from_images() -> str:
    """Extract text from image files using improved OCR techniques."""
    if not os.path.exists(PRELOADED_DOCS_FOLDER):
        return ""
    
    all_text = ""
    image_files = [
        f for f in os.listdir(PRELOADED_DOCS_FOLDER)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    
    for image_file in image_files:
        image_path = os.path.join(PRELOADED_DOCS_FOLDER, image_file)
        try:
            # Improved image preprocessing for better OCR
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding to handle different lighting conditions
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Noise removal
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # OCR with improved settings
            custom_config = r'--oem 3 --psm 6'  # Assume a single uniform block of text
            text = pytesseract.image_to_string(opening, config=custom_config)
            
            all_text += f"\n\n--- Image: {image_file} ---\n{text.strip()}"
        except Exception as e:
            logger.error(f"Error processing image {image_file}: {str(e)}")
    
    return all_text.strip()


def transcribe_videos() -> str:
    """Transcribe audio from video files with improved reliability."""
    if not os.path.exists(PRELOADED_DOCS_FOLDER):
        return ""
    
    all_text = ""
    video_files = [
        f for f in os.listdir(PRELOADED_DOCS_FOLDER)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]
    
    for video_file in video_files:
        video_path = os.path.join(PRELOADED_DOCS_FOLDER, video_file)
        # Create a temporary file that will be cleaned up automatically
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            audio_path = temp_file.name
        
        try:
            # Extract audio with error handling
            try:
                ffmpeg.input(video_path).output(
                    audio_path, format="wav", acodec="pcm_s16le", ac=1, ar="16k"
                ).run(overwrite_output=True, quiet=True)
            except ffmpeg.Error as e:
                logger.error(f"FFmpeg error processing {video_file}: {e.stderr.decode() if e.stderr else str(e)}")
                continue
            
            # Use a more appropriate whisper model based on file size
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
            
            # Choose model size based on file size
            model_size = "tiny" if file_size < 10 else "base"
            logger.info(f"Using {model_size} model for {video_file} ({file_size:.2f} MB)")
            
            model = whisper.load_model(model_size, device="cpu")
            
            # Transcribe with appropriate parameters
            result = model.transcribe(
                audio_path,
                fp16=False,
                language="en"  # Set the expected language if known
            )
            
            all_text += f"\n\n--- Video: {video_file} ---\n{result['text']}"
        except Exception as e:
            logger.error(f"Error processing video {video_file}: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    return all_text.strip()


def extract_text_from_txt() -> str:
    """Extract text from plain text files with improved encoding handling."""
    if not os.path.exists(PRELOADED_DOCS_FOLDER):
        return ""
    
    all_text = ""
    text_files = [f for f in os.listdir(PRELOADED_DOCS_FOLDER) if f.endswith(".txt")]
    
    for text_file in text_files:
        text_path = os.path.join(PRELOADED_DOCS_FOLDER, text_file)
        try:
            # Try multiple encodings to handle different file types
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(text_path, "r", encoding=encoding) as file:
                        content = file.read()
                    break  # If successful, break the loop
                except UnicodeDecodeError:
                    continue
            
            if content is not None:
                all_text += f"\n\n--- Document: {text_file} ---\n{content.strip()}"
            else:
                logger.warning(f"Could not decode {text_file} with any of the attempted encodings")
        except Exception as e:
            logger.error(f"Error processing text file {text_file}: {str(e)}")
    
    return all_text.strip()


def create_vectorstore(text: str) -> FAISS:
    """
    Creates a FAISS vector store from the input text with optimized parameters.
    """
    if not text:
        logger.warning("Attempted to create vectorstore with empty text")
        return None
    
    # Use optimized parameters for better retrievals
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Larger chunks for more context
        chunk_overlap=300,  # More overlap to avoid missing context at boundaries
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # More granular separation
    )
    
    # Split text into chunks
    try:
        texts = text_splitter.split_text(text)
        logger.info(f"Text split into {len(texts)} chunks")
        
        # Log the first few chunks for debugging
        for i, chunk in enumerate(texts[:3]):
            logger.debug(f"Sample chunk {i + 1}: {chunk[:100]}...")
        
        # Create documents
        documents = [Document(page_content=t) for t in texts]
        
        # Create and return the vector store with error handling
        try:
            vectorstore = FAISS.from_documents(documents, embeddings)
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        return None


# ---------------- AI Tutor Function ----------------

def answer_question(state: TutorState, user_question: str) -> str:
    """
    Uses GPT to answer questions based on vectorized content with improved relevance.
    """ 
    # Keep a limited history to maintain context without excessive token usage
    state["chat_history"] = state["chat_history"][-8:]  # Last 8 exchanges
    
    if not state["vectorstore"]:
        return "I'm sorry, but no study materials have been loaded yet. Please check that documents are in the 'files' folder."
    
    try:
        # Improved search with MMR to get diverse but relevant results
        try:
            # First try with MMR (Maximum Marginal Relevance) to get diverse results
            relevant_docs = state["vectorstore"].max_marginal_relevance_search(
                user_question,
                k=5,  # Get top 5 relevant chunks
                fetch_k=10,  # Fetch 10 and select most diverse 5
                lambda_mult=0.7  # Balance between relevance and diversity
            )
        except AttributeError:
            # Fall back to standard similarity search if MMR not available
            relevant_docs = state["vectorstore"].similarity_search(
                user_question,
                k=4  # Get top 4 most relevant chunks
            )
        
        # Combine relevant texts
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Format chat history for better context awareness
        chat_history = ""
        for entry in state["chat_history"]:
            chat_history += f"{entry}\n"
        
        # Improved prompt with better instructions and more context
        prompt = f"""
You are an AI Tutor for students. Your goal is to answer questions based on the provided study materials.

**Instructions:**
1. If the answer is found in the study materials, provide a detailed, educational response.
2. If the question is related to but not directly answered in the materials, use your knowledge to provide relevant information while noting this fact.
3. If the question is completely unrelated to the study materials, politely redirect the student by saying: "I'm sorry, but I can only answer questions based on the provided study material. Please ask something related to the content."
4. Give explanations at an appropriate level for a student, with clear examples where helpful.
5. If appropriate, mention the specific document or section where the information comes from.

**Study Material Context:**
{context}

**Previous Conversation:**
{chat_history}

**Student's Question:**
{user_question}
"""
        
        # Get response with timeout handling
        response = llm.invoke(prompt)
        
        # Update chat history
        state["chat_history"].append(f"User: {user_question}\nAI: {response.content}")
        return response.content
    
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return "I'm sorry, I encountered an error while processing your question. Please try again later."


# ---------------- FastAPI Application ----------------

app = FastAPI(title="AI Tutor API", description="API for an AI Tutor system using pre-loaded documents")

# Add CORS middleware with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# Pydantic models for request and response
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    chat_history: List[str]


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup and load pre-existing documents."""
    logger.info("Starting AI Tutor API...")
    logger.info(f"Looking for documents in {PRELOADED_DOCS_FOLDER}...")
    
    # Count files by type to log
    pdf_count = len([f for f in os.listdir(PRELOADED_DOCS_FOLDER) if f.lower().endswith(".pdf")])
    img_count = len([f for f in os.listdir(PRELOADED_DOCS_FOLDER) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    vid_count = len([f for f in os.listdir(PRELOADED_DOCS_FOLDER) if f.lower().endswith((".mp4", ".avi", ".mov"))])
    txt_count = len([f for f in os.listdir(PRELOADED_DOCS_FOLDER) if f.lower().endswith(".txt")])
    
    logger.info(f"Found {pdf_count} PDFs, {img_count} images, {vid_count} videos, and {txt_count} text files.")
    
    # Load and process files
    await process_files_async()
    
    # Verify the content was loaded
    if tutor_state["vectorstore"] is None:
        logger.warning("No content was successfully vectorized. Check the files in the directory.")
    else:
        logger.info("System initialized successfully and ready to answer questions.")


@app.get("/")
def read_root():
    """Root endpoint to check if the API is running."""
    return {
        "message": "AI Tutor API is running",
        "status": "ok",
        "documents_loaded": tutor_state["vectorstore"] is not None
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question_endpoint(request: QuestionRequest):
    """
    Endpoint to ask a question based on loaded study materials.
    """
    if not tutor_state.get("vectorstore"):
        # Try to reprocess files if vectorstore is empty
        await process_files_async()
        
        # Check again after processing
        if not tutor_state.get("vectorstore"):
            raise HTTPException(
                status_code=400, 
                detail="No study materials could be processed. Please ensure files exist in the 'files' directory."
            )
    
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
    tutor_state["chat_history"] = []
    return tutor_state["chat_history"]


@app.get("/status")
async def get_system_status():
    """
    Endpoint to check system status and loaded materials.
    """
    # Count files by type
    file_counts = {
        "pdf": 0,
        "image": 0,
        "video": 0, 
        "text": 0,
        "total": 0
    }
    
    # Check files folder
    if os.path.exists(PRELOADED_DOCS_FOLDER):
        for file in os.listdir(PRELOADED_DOCS_FOLDER):
            ext = os.path.splitext(file)[1].lower()
            if ext == ".pdf":
                file_counts["pdf"] += 1
            elif ext in [".jpg", ".jpeg", ".png"]:
                file_counts["image"] += 1
            elif ext in [".mp4", ".avi", ".mov"]:
                file_counts["video"] += 1
            elif ext == ".txt":
                file_counts["text"] += 1
            file_counts["total"] += 1
    
    return {
        "status": "running",
        "files": file_counts,
        "vectorstore_loaded": tutor_state["vectorstore"] is not None,
        "chat_history_length": len(tutor_state["chat_history"])
    }