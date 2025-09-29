import os
from dotenv import load_dotenv

# --- RAG Libraries ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings # <--- CORRECT IMPORT
from langchain_openai import ChatOpenAI 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Configuration and Initialization ---

PDF_FILE_PATH = "document.pdf" 
COLLECTION_NAME = "pdf_summarizer_memory" 

load_dotenv()
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY") 

# 1. Initialize the Hugging Face Embedding Model
# This model is downloaded and runs locally (CPU) and is very fast.
# It requires no API key, which resolves the 'ValueError' by bypassing cloud API auth.
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", # A lightweight, high-performance model
    model_kwargs={'device': 'cpu'} # Ensure it runs on your CPU
)

# 2. Document Processing Pipeline (Corrected)
def process_document_for_rag(file_path: str):
    """
    Loads a PDF, splits it into chunks, and stores the vectors in ChromaDB.
    """
    print(f"--- 1/3: Loading and Splitting {file_path} ---")
    
    loader = PyPDFLoader(file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs = loader.load_and_split(text_splitter=text_splitter)
    print(f"Loaded and split into {len(docs)} chunks.")
    
    # --- 2/3: Creating Vector Store (Embedding and Persistence) ---
    print("--- 2/3: Creating Vector Store (RAG Memory) ---")
    
    # The 'embeddings' object is now the local HuggingFace model.
    # Chroma successfully uses it to generate vectors and save them.
    db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        collection_name=COLLECTION_NAME,
        persist_directory="./chroma_db"
    )
    
    print("Vector store created successfully in ./chroma_db")
    return db

# 3. Create the Conversational RAG Chain (No change in logic)
def create_conversational_chain(vector_db: Chroma):
    """Connects the vector database to the LLM to answer questions."""
    # ... (Rest of the function remains the same, using ChatOpenAI/DeepSeek)
    llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=openrouter_api_key,
        model_name="deepseek/deepseek-chat-v3.1",
        temperature=0.1
    )
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff" 
    )
    print("RAG Chain is ready to use!")
    return qa_chain

# --- Main Interaction Loop ---
if __name__ == "__main__":
    # The very first time you run this, it will download the embedding model (about 400MB).
    # Please ensure you have a 'document.pdf' file ready!
    try:
        qa_db = process_document_for_rag(PDF_FILE_PATH)
        qa_chain = create_conversational_chain(qa_db)

        print("\n--- PDF Summarizer Agent Initialized (Type 'exit' to quit) ---")
        while True:
            user_input = input("Ask about the PDF: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            result = qa_chain.invoke({"question": user_input})
            print(f"\nAgent Answer: {result['answer']}\n")
            
    except FileNotFoundError:
        print(f"\nERROR: Please create a file named '{PDF_FILE_PATH}' in your project directory.")
    except Exception as e:
        # Catch any remaining errors, especially API or missing dependency errors
        print(f"\nAn unexpected error occurred: {e}")
        print("Please ensure you have all libraries installed and your OpenRouter API key is valid.")