# Add these imports at the top
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Add this after initializing OpenAI
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Modify the TutorState class
class TutorState(TypedDict):
    extracted_content: str
    chat_history: List[str]
    vectorstore: Any  # Add this line

# Initialize vectorstore in state
tutor_state: TutorState = {
    "extracted_content": "", 
    "chat_history": [],
    "vectorstore": None
}

# Add this function to create the vector store
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
    
    # Create documents
    documents = [Document(page_content=t) for t in texts]
    
    # Create and return the vector store
    return FAISS.from_documents(documents, embeddings)

# Modify the load_study_materials function
@app.on_event("startup")
def load_study_materials():
    """
    On startup, load study materials and create vector store
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

# Modify the answer_question function
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
You are an AI Tutor for school students. Base your answer on these relevant excerpts from the study materials:

{context}

Previous Conversation:
{chat_context}

Student's Question:
{user_question}

Please provide a detailed, educational answer based on the relevant content above.
If the question cannot be answered from the provided content, say so.
"""
    
    response = llm.invoke(prompt)
    state["chat_history"].append(f"User: {user_question}\nAI: {response.content}")
    return response.content
