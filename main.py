# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import logging
import re
import uuid
from starlette.concurrency import run_in_threadpool
import asyncio
from typing import Optional
from fastapi import Query
from datetime import datetime, date
import pandas as pd
from contextlib import asynccontextmanager
from typing import List


# --- Load environment variables from .env file ---
from dotenv import load_dotenv
load_dotenv()

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- Define Constants ---
UPLOAD_DIR = "./data"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "Financial_Reports" # Consistent collection name
SESSION_COOKIE_NAME = "rag_session_id" # Define your cookie name

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# --- CORS Middleware ---
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:567m")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gemma3:27b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

# --- Global Variables for RAG Components ---
embeddings = None
llm = None
vector_store = None
retriever = None

# --- Keep your original ChatContext class (no Firebase code in it) ---
class ChatContext:
    """
    Manages the conversation history for an LLM, storing user queries and assistant answers.
    Handles context window trimming based on the number of conversation turns.
    """
    def __init__(self, max_turns: int = 5):
        self.history = []
        self.max_messages = max_turns * 2
        logger.info("New ChatContext instance created.")

    def add_query(self, query: str):
        self.history.append({"role": "user", "content": query})
        self._trim_history()
        logger.info(f"Added query: '{query[:40]}...'")

    def add_answer(self, answer: str):
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()
        logger.info(f"Added answer: '{answer[:40]}...'")

    def _trim_history(self):
        if len(self.history) > self.max_messages:
            self.history = self.history[len(self.history) - self.max_messages:]
            logger.info(f"History trimmed. Current messages: {len(self.history)}")

    def get_context(self) -> str:
        formatted_history = []
        for i in range(0, len(self.history), 2):
            user_msg = self.history[i]
            if i + 1 < len(self.history):
                assistant_msg = self.history[i+1]
                formatted_history.append(f"Question: {user_msg['content']}\nAnswer: {assistant_msg['content']}")
            else:
                formatted_history.append(f"Question: {user_msg['content']}")

        if not formatted_history:
            return ""
        return "Previous conversations:\n" + "\n".join(formatted_history)

    def reset_context(self):
        self.history = []
        logger.info("Conversation history reset.")

    def _clean_llm_output(self, text: str) -> str:
        """
        Removes content enclosed in <think>...</think> tags.
        """
        # The regex pattern matches <think> followed by any characters (non-greedy)
        # up to </think>. re.DOTALL ensures it matches across newlines.
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned_text.strip()

# --- In-memory cache for ChatContext instances ---
# This dictionary will hold a ChatContext object for each active session ID.
active_chat_sessions: dict[str, ChatContext] = {}

def get_chat_session_by_id(session_id: str) -> ChatContext:
    """
    Retrieves or creates a ChatContext instance for the given session_id from in-memory cache.
    """
    if session_id not in active_chat_sessions:
        active_chat_sessions[session_id] = ChatContext()
        logger.info(f"Created new in-memory chat session for ID: {session_id}")
    return active_chat_sessions[session_id]


# --- Your existing extract_sources_from_raw_chunks function ---
def extract_sources_from_raw_chunks(raw_chunks_content: list[str]) -> set[str]:
    sources_set = set()
    source_pattern = re.compile(r'^\[Source:\s*(.+?)\]')
    for chunk_content in raw_chunks_content:
        match = source_pattern.match(chunk_content)
        if match:
            source_filename = match.group(1).strip()
            sources_set.add(source_filename)
    return sources_set

        # Format them for the frontend
def format_period_for_frontend(period_str):
    """Convert 'Q3_2024-Q4_2024' to 'Q4_2024 vs Q3_2024'"""
    if '-' in period_str:
        start, end = period_str.split('-')
        return f"{end} vs {start}"
    return period_str

def convert_frontend_to_csv_format(frontend_period):
    """Convert 'Q1_2025 vs Q4_2024' to 'Q4_2024-Q1_2025'"""
    if ' vs ' in frontend_period:
        end, start = frontend_period.split(' vs ')
        return f"{start}-{end}"
    return frontend_period

def is_sentiment_fetch_allowed():
    """
    Check if sentiment data fetching is allowed based on current date and quarterly cutoffs.
    
    Returns:
        tuple: (is_allowed: bool, message: str, next_allowed_date: str)
    """
    CSV_FILE = "data/sector_sentiment_analysis_summary.csv"
    if not os.path.exists(CSV_FILE):
        # Default periods if CSV doesn't exist
        first_transcript_year = 2024
        first_transcript_quarter = 3
        second_transcript_year = 2024
        second_transcript_quarter = 4
        print(f"\nCSV file '{CSV_FILE}' not found. Using default periods:")
        print(f"First transcript: Q{first_transcript_quarter} {first_transcript_year}")
        print(f"Second transcript: Q{second_transcript_quarter} {second_transcript_year}")
    else:
        # Read existing CSV and determine the next period
        try:
            df = pd.read_csv(CSV_FILE)
            
            # Look for period information in the CSV
            # This assumes the CSV has columns or data that indicates the periods analyzed
            # You might need to adjust this based on your actual CSV structure
            
            # Method 1: If periods are stored in column names or headers
            # Look for patterns like "Q4_2024-Q1_2025" in column names
            period_pattern = r'Q(\d)_(\d{4})-Q(\d)_(\d{4})'
            periods_found = []
            
            # Check column names for period information
            for col in df.columns:
                if col != 'Sector':  # Skip the Sector column
                    match = re.search(period_pattern, str(col))
                    if match:
                        # Extract both periods from the column name
                        first_quarter, first_year = int(match.group(1)), int(match.group(2))
                        second_quarter, second_year = int(match.group(3)), int(match.group(4))
                        periods_found.append(((first_year, first_quarter), (second_year, second_quarter)))
            
            if periods_found:
                # Find the latest period pair (based on the second period in each pair)
                latest_period_pair = max(periods_found, key=lambda x: (x[1][0], x[1][1]))
                latest_first_period, latest_second_period = latest_period_pair
                
                # The next analysis should start from the second period of the latest analysis
                first_transcript_year = latest_second_period[0]
                first_transcript_quarter = latest_second_period[1]
                
                # Calculate the next quarter for the second transcript
                if first_transcript_quarter == 4:
                    second_transcript_year = first_transcript_year + 1
                    second_transcript_quarter = 1
                else:
                    second_transcript_year = first_transcript_year
                    second_transcript_quarter = first_transcript_quarter + 1
        except Exception as e:
            print(f"\nError reading CSV file: {e}")
            print("Using default periods:")
            first_transcript_year = 2024
            first_transcript_quarter = 4
            second_transcript_year = 2025
            second_transcript_quarter = 1
            print(f"First transcript: Q{first_transcript_quarter} {first_transcript_year}")
            print(f"Second transcript: Q{second_transcript_quarter} {second_transcript_year}")

    today = date.today()

    def get_cutoff_date(quarter, year):
        cutoff_dates = {
            1: date(year, 4, 15),   # Q1 data available Apr 15
            2: date(year, 7, 15),   # Q2 data available Jul 15  
            3: date(year, 10, 15),  # Q3 data available Oct 15
            4: date(year + 1, 1, 15)  # Q4 data available Jan 15 of next year
        }
        return cutoff_dates[quarter]
    
    first_transcript_cutoff = get_cutoff_date(first_transcript_quarter, first_transcript_year)
    second_transcript_cutoff = get_cutoff_date(second_transcript_quarter, second_transcript_year)
    
    # Check if first transcript data is available
    if today < first_transcript_cutoff:
        return False, f"Q{first_transcript_quarter} {first_transcript_year} sentiment data not available until {first_transcript_cutoff.strftime('%B %d, %Y')}", first_transcript_cutoff.strftime("%B %d, %Y")
    
    # Check if second transcript data is available
    if today < second_transcript_cutoff:
        return False, f"Q{second_transcript_quarter} {second_transcript_year} sentiment data not available until {second_transcript_cutoff.strftime('%B %d, %Y')}", second_transcript_cutoff.strftime("%B %d, %Y")
    
    # Both transcript periods have data available
    return True, "Current period data available", ""

# --- Import your ingestion service ---
try:
    from ingest_services import ingest_documents_to_chroma
except ImportError as e:
    logger.error(f"Error: Could not import 'ingest_services.py'. Make sure the file exists and functions are correctly named. Error: {e}")
    ingest_documents_to_chroma = None

# --- LangChain Imports for RAG Querying ---
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

@asynccontextmanager
async def lifespan(app: FastAPI): # <--- NEW FUNCTION DEFINITION
    """
    Handles startup and shutdown events for the FastAPI application.
    Our RAG components will be initialized here.
    """
    global embeddings, llm, vector_store, retriever # Declare globals to modify them

    logger.info("FastAPI application lifespan: Startup phase initiated.")
    try:
        # --- RAG Components Initialization (Startup Logic) ---
        # This is the SAME initialization code that was previously at the top-level
        # and in your old `startup_event` function.
        logger.info(f"Attempting to initialize RAG components...")
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        logger.info(f"OllamaEmbeddings initialized.")

        llm = ChatOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            num_ctx=8192 * 2
        )
        logger.info(f"ChatOllama initialized.")

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=DB_DIR,
        )
        logger.info(f"ChromaDB vector_store initialized from {DB_DIR} with collection: {COLLECTION_NAME}.")

        retriever = vector_store.as_retriever(search_kwargs={"k": 15})
        logger.info("Retriever created successfully.")

    except Exception as e:
        logger.error(f"Error during RAG components initialization on startup: {e}")
        embeddings = None
        llm = None
        vector_store = None
        retriever = None

    yield # <--- This line is crucial: it hands control to FastAPI to start serving requests

    # --- Shutdown Logic (runs when the app is shutting down) ---
    logger.info("FastAPI application lifespan: Shutdown phase initiated.")
    # Any cleanup code can go here

app = FastAPI(
    title="RAG Application API",
    description="API for document upload, ingestion, and RAG services.",
    version="0.1.0",
    lifespan=lifespan
)

logger.info(f"OLLAMA_BASE_URL as seen by FastAPI: {OLLAMA_BASE_URL}")

origins = [
    "http://localhost",
    "http://localhost:8000",
    # Add your frontend's actual URL here when deployed, e.g., "https://your-frontend-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # Crucial for cookies to be sent/received
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

embeddings = None
llm = None
vector_store = None
retriever = None

# --- Define the System Prompt Content ---
SYSTEM_PROMPT_CONTENT = """
You are a highly accurate financial research assistant. Your primary goal is to extract and present precise numerical data and factual information from company annual reports. You are able to leverage previous conversation turns to provide more coherent and contextually relevant answers.

**STRICT RULES FOR ANSWERING:**
0.  **SOURCE RELEVANCE:** Each piece of information in "The Data (Newly Retrieved Relevant Information)" is tagged with its source (e.g., `(Source: some_document_id)`, `(Source: Q4_2023_Report)`, `(Source: Financial_Statement_Page_7)`). Your primary task is to identify and use **only** the source(s) that are directly relevant to the user's current question. If a source tag itself does not explicitly state the company name, year, or other direct identifiers, you MUST infer its relevance from the **content of the data chunk itself** in conjunction with the user's query and previous conversation context. Your answer MUST then be derived **solely** from the content of these identified relevant sources.
1.  **ANSWER ONLY FROM SELECTED RELEVANT SOURCES:** You MUST only use the information from the **identified relevant source(s) within "The Data (Newly Retrieved Relevant Information)"** and/or 'Previous Conversation Context' sections below. Do NOT use any external knowledge, pre-trained biases, or make assumptions.
2.  **NUMERICAL ACCURACY IS PARAMOUNT:** When extracting numbers (e.g., financial figures, percentages, dates, quantities), ensure absolute precision. Include units (e.g., SGD million, %, units, years) and specific values as presented in the source.
3.  **TABLE INTERPRETATION:**
    * **Table Structure:** You may receive tables presented in **Markdown format** (structured with characters like `|` and `-`) or, in **CSV format** (using `,` to separate values). Interpret the respective formatting to accurately understand column headers, row relationships, and individual cell values.
    * **Contextual Understanding:** Use table headers, row labels, and surrounding text to correctly interpret the meaning of numerical values. For example, if a table shows "Revenue" and "Profit", clearly distinguish between them.
    * **Aggregation/Calculation:** If a question requires simple aggregation or calculation (e.g., sum, difference, average) *that can be directly derived from numbers present in the provided data*, perform it. State explicitly if a calculation was performed.
4.  **CONSISTENCY:** Ensure all facts, figures, and terminology are consistent with the provided data.
5.  **CONFIDENCE & UNCERTAINTY:** If the information required to answer the question, especially numerical data, is not explicitly present, is ambiguous, or cannot be precisely derived from "The Data", you MUST state: "I am unable to find a precise answer based on the provided information." Do NOT attempt to guess or estimate.
6.  **FORMATTING:** Present numerical answers clearly, often in a bulleted or tabular format if multiple data points are relevant.

--------------------
**Previous Conversation Context (for continuity, if available):**
{previous_conversation}

--------------------
**The Data (Newly Retrieved Relevant Information):**
{retrieved_data}
"""

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return HTMLResponse("""
    <h1>Welcome to the RAG Application API!</h1>
    <p>Visit <a href="/docs">/docs</a> for the interactive API documentation.</p>
    """)

@app.post("/uploadfiles/", summary="Upload Multiple PDF Documents")
async def upload_multiple_pdfs(files: List[UploadFile] = File(...)): #files: List[UploadFile] = File(...)
    """
    Upload multiple PDF files at once.
    Returns a summary of successful and failed uploads.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    
    successful_uploads = []
    failed_uploads = []
    
    for file in files:
        try:
            # Validate file type
            if not file.filename.endswith(".pdf"):
                failed_uploads.append({
                    "filename": file.filename,
                    "error": "Only PDF files are allowed."
                })
                continue
            
            # Check if file already exists and handle duplicates
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            
            # Save the file
            with open(file_path, "wb") as buffer:
                # Reset file pointer to beginning
                await file.seek(0)
                content = await file.read()
                buffer.write(content)
            
            successful_uploads.append({
                "filename": file.filename,
                "path": file_path
            })
            logger.info(f"Successfully uploaded and saved: {file_path}")
            
        except Exception as e:
            failed_uploads.append({
                "filename": file.filename,
                "error": f"Could not save file: {str(e)}"
            })
            logger.error(f"Error saving file {file.filename}: {e}")
    
    # Prepare response
    response_data = {
        "total_files": len(files),
        "successful_uploads": len(successful_uploads),
        "failed_uploads": len(failed_uploads),
        "successful_files": [upload["filename"] for upload in successful_uploads],
        "failed_files": failed_uploads
    }
    
    # Determine response status
    if len(successful_uploads) == 0:
        raise HTTPException(
            status_code=400, 
            detail={
                "message": "No files were uploaded successfully.",
                **response_data
            }
        )
    elif len(failed_uploads) > 0:
        # Partial success - return 207 Multi-Status or 200 with warnings
        response_data["message"] = f"{len(successful_uploads)} files uploaded successfully, {len(failed_uploads)} failed. Trigger /ingest to process uploaded files."
        return response_data
    else:
        # Complete success
        response_data["message"] = f"All {len(successful_uploads)} files uploaded successfully. Trigger /ingest to process."
        return response_data

@app.post("/uploadfile/", summary="Upload PDF Document")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Successfully uploaded and saved: {file_path}")
        return {"filename": file.filename, "message": "File uploaded successfully. Trigger /ingest to process."}
    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

@app.post("/ingest/", summary="Process and Ingest Documents")
async def ingest_documents():
    # --- CRUCIAL ADDITION: Declare global variables before re-assigning them ---
    global embeddings, llm, vector_store, retriever 

    if ingest_documents_to_chroma is None:
        logger.error("Ingestion service not loaded. Cannot perform ingestion.")
        raise HTTPException(status_code=500, detail="Ingestion service is unavailable.")
    try:
        logger.info("Ingestion request received. Starting document processing...")
        # AWAITING run_in_threadpool will offload the potentially blocking call
        # to a separate thread, allowing the main event loop to remain free.
        await run_in_threadpool(
            ingest_documents_to_chroma,
            global_vector_store=vector_store, # Pass the global vector_store
            global_embeddings=embeddings,   # Pass the global embeddings
            data_dir=UPLOAD_DIR,            # Pass your UPLOAD_DIR constant
            # Add other parameters like use_adobe_api, reset_collection if needed
            # use_adobe_api=False,
            # reset_collection=False
        )
        logger.info("Ingestion process completed.")
        
        logger.info("Calling initialize_rag_components after ingestion to refresh vector store view.")
        
        # Re-use embeddings/LLM if they already exist, no need to re-create if they don't change
        # These checks are good, but still need `global` if they might be reassigned
        if embeddings is None: # This assignment needs `global` if embeddings was None
            embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
            logger.info("Embeddings re-initialized (was None).")

        if llm is None: # This assignment needs `global` if llm was None
            llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1, num_ctx=8192*2)
            logger.info("LLM re-initialized (was None).")

        # --- IMPORTANT: Create a NEW Chroma instance to pick up new data ---
        # This assignment *definitely* needs `global`
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=DB_DIR,
        )
        logger.info(f"ChromaDB vector_store re-initialized from {DB_DIR} after ingestion.")

        # This assignment *definitely* needs `global`
        retriever = vector_store.as_retriever(search_kwargs={"k": 15})
        logger.info("Retriever re-created after ingestion.")

        return {"message": "Ingestion process completed and RAG system refreshed. Check logs for details."}
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Error during ingestion: {e}")

# --- UPDATED: RAG Query Endpoint with Cookie-based Session Management ---
@app.post("/query/", summary="Query the RAG System")
async def query_rag(request_body: QueryRequest, http_request: Request, response: Response):
    """
    Queries the RAG system with a given text query, using the specific logic and prompt
    from your ask_with_params.py script, and maintains conversation history per user
    using a session cookie.
    """
    # 1. Session Identification: Get session_id from cookie or generate a new one
    session_id = http_request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            httponly=True, # Prevent JavaScript access to the cookie
            samesite="Lax", # Or "None" if cross-site, but requires secure=True
            secure=False # Set to True if deployed with HTTPS
            # You can also add 'expires' for persistent cookies (e.g., max_age=3600*24*7 for 7 days)
            # Default is a "session cookie" which disappears when browser closes
        )
        logger.info(f"New session created and cookie set for ID: {session_id}")
    else:
        logger.info(f"Existing session found with ID: {session_id}")

    # 2. Get/Create ChatContext for this session ID
    chat_session = get_chat_session_by_id(session_id)

    query = request_body.query

    if retriever is None or llm is None:
        logger.error("RAG components (retriever or LLM) are not initialized. Cannot process query.")
        raise HTTPException(status_code=500, detail="RAG system is unavailable. Check server logs for initialization errors.")

    logger.info(f"Received query for session {session_id}: {query}")

    try:
        # 4. Perform RAG Retrieval
        retrieved_docs = retriever.invoke(query)
        retrieved_documents_content = [doc.page_content for doc in retrieved_docs]
        logger.info(f"retrieved_documents_content: {retrieved_documents_content[:200]}...")
        source_set = extract_sources_from_raw_chunks(retrieved_documents_content)
        source_list = list(source_set)

        # 5. Get current conversation context string from the session
        prev_convo_context_string = chat_session.get_context()
        logger.info(f"Previous conversation context for session {session_id}: {prev_convo_context_string[:200]}...")

        # 6. Prepare RAG context (now named retrieved_data_string) for the LLM
        retrieved_data_string = "\n\n---\n\n".join(retrieved_documents_content)
        # 7. Format the system prompt with the separated contexts
        formatted_system_prompt = SYSTEM_PROMPT_CONTENT.format(
            previous_conversation=prev_convo_context_string if prev_convo_context_string else "No previous conversation.",
            retrieved_data=retrieved_data_string
        )

        # 8. Create structured chat messages for the LLM
        messages = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=formatted_system_prompt),
                HumanMessage(content=query),
            ]
        )

        # 9. Invoke the LLM
        response_obj = llm.invoke(messages.format_messages())
        response_content = response_obj.content

        logger.info(f"Generated response for query: '{query}' for session {session_id}")
        logger.info(f"Sources: {source_list}")
        source_string = ", ".join(source_list)

        # 10. Add the LLM response to chat session history
        # Add current query to chat session history
        chat_session.add_query(query)
        response_content = chat_session._clean_llm_output(response_content)   # Clean the LLM output
        chat_session.add_answer(response_content)

        return {"query": query, "response": response_content, "sources": source_string}

    except Exception as e:
        logger.error(f"Error during RAG query for '{query}' (session {session_id}): {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}. Ensure Ollama server is running and accessible and models ({OLLAMA_EMBEDDING_MODEL}, {OLLAMA_LLM_MODEL}) are pulled.")

# --- NEW: Endpoint to reset chat history for a specific session ---
@app.post("/reset_chat/", summary="Reset Chat Conversation History for a Session")
async def reset_chat_history(http_request: Request, response: Response):
    """
    Resets the chat history for the current session identified by a cookie.
    """
    session_id = http_request.cookies.get(SESSION_COOKIE_NAME)

    if not session_id:
        raise HTTPException(status_code=400, detail="No active session cookie found to reset.")

    if session_id in active_chat_sessions:
        active_chat_sessions[session_id].reset_context()
        del active_chat_sessions[session_id]
        # Remove the cookie from the browser as well
        response.delete_cookie(key=SESSION_COOKIE_NAME)
        logger.info(f"In-memory chat history and cookie reset for session {session_id} via API call.")
        return {"message": f"In-memory chat history has been reset for session {session_id}."}
    else:
        raise HTTPException(status_code=404, detail=f"No active session found for ID: {session_id}.")


@app.get("/sentiment/display/", summary="Display Existing Sentiment Data for Selected Period")
async def display_sentiment_data(period: Optional[str] = Query(None, description="Specific period to display (e.g., 'Q1_2025 vs Q4_2024')")):
    try:
        csv_path = "data/sector_sentiment_analysis_summary.csv"
        
        if not os.path.exists(csv_path):
            return {"status": "error", "message": "No sentiment data found."}

        import pandas as pd
        
        # Read CSV with all columns as strings to prevent parsing
        df = pd.read_csv(csv_path, dtype=str)
        
        if df.empty:
            return {"status": "error", "message": "Sentiment data file is empty."}
        
        # Get CSV column names (excluding 'Sector')
        csv_columns = [col for col in df.columns if col != 'Sector']
        
        # Determine which period to use
        if period:
            # Convert frontend format to CSV format for lookup
            csv_format_period = convert_frontend_to_csv_format(period)
            
            # Check if the converted period exists in the columns
            if csv_format_period not in csv_columns:
                # Convert available CSV periods to frontend format for error message
                available_frontend_periods = [format_period_for_frontend(col) for col in csv_columns]
                return {
                    "status": "error", 
                    "message": f"Period '{period}' not found. Available periods: {available_frontend_periods}"
                }
            selected_csv_period = csv_format_period
            selected_frontend_period = period
        else:
            # Default to latest period (last column)
            selected_csv_period = csv_columns[-1]
            selected_frontend_period = format_period_for_frontend(selected_csv_period)
        
        sentiment_data = {}
        for _, row in df.iterrows():
            sector = row['Sector']
            sentiment_value = row[selected_csv_period]
            
            # Handle NaN/None values
            if pd.isna(sentiment_value) or sentiment_value == 'nan':
                sentiment_data[sector] = "(N/A, 0)"
            else:
                # It's already a string now, just clean it
                str_value = str(sentiment_value).strip()
                # Remove quotes if pandas added them
                if str_value.startswith('"') and str_value.endswith('"'):
                    str_value = str_value[1:-1]
                sentiment_data[sector] = str_value
        
        # Convert all available CSV periods to frontend format for response
        available_frontend_periods = [format_period_for_frontend(col) for col in csv_columns]
        
        return {
            "status": "success",
            "data": {
                "sectors": sentiment_data,
                "period": selected_frontend_period,
                "total_sectors": len(sentiment_data),
                "available_periods": available_frontend_periods
            },
            "message": f"Sentiment data for {selected_frontend_period} loaded successfully"
        }
    
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}

# --- NEW: Sentiment Analysis Endpoint ---
@app.get("/sentiment/get_newest/", summary="Fetch Sentiment Analysis Data")
async def get_sentiment_data():
    """
    Fetches sentiment analysis data by running the sentiment analysis logic.
    Updates the CSV file with new sentiment data for all 11 sectors.
    """
    try:
        logger.info("Fetching sentiment analysis data...")
        is_allowed, status_message, next_date = is_sentiment_fetch_allowed()
        
        if not is_allowed:
            return {
                "status": "error",
                "message": f"{status_message}. Next update available on {next_date}",
                "data": None
            }
        # Import the sentiment analysis function
        from full_sentiment import get_sentiment
        import pandas as pd
        
        # Run the sentiment analysis (this will update the CSV file)
        await asyncio.get_event_loop().run_in_executor(
            None, get_sentiment
        )
        
        # Read the updated CSV file to return the latest data
        # Adjust the path to match your CSV file location
        csv_path = "data/sector_sentiment_analysis_summary.csv"  # Update this path as needed
        df = pd.read_csv(csv_path, dtype=str)
        period_columns = [col for col in df.columns if col != 'Sector'] #list of all period columns except 'Sector'


        available_periods = [format_period_for_frontend(col) for col in period_columns] #formatted list of periods
        
        # Convert DataFrame to dictionary format for JSON response
        sentiment_data = {}
        for _, row in df.iterrows():
            sector = row['Sector']
            # Get the latest column (assuming it's the last column)
            latest_period = df.columns[-1] #gets the last column name
            sentiment_value = row[latest_period] #gets the value for the latest period
            sentiment_data[sector] = sentiment_value #gets the sentiment value for each sector

        formatted_latest_period = format_period_for_frontend(latest_period)
        
        return {
            "status": "success",
            "data": {
                "sectors": sentiment_data,
                "period": formatted_latest_period,
                "total_sectors": len(sentiment_data),
                "availablePeriods": available_periods
            },
            "message": "Sentiment analysis completed and CSV updated successfully"
        }
        
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to import sentiment analysis module: {str(e)}"
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment data CSV file not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}"
        )
@app.get("/sentiment/refresh/", summary="Refresh Sentiment Analysis Data")
async def refresh_sentiment_data():
    """
    Refresh sentiment analysis by recalculating the latest period data
    and replacing the last column in the CSV file.
    """
    try:
        logger.info("Refreshing sentiment analysis data...")
        # Import the refresh sentiment function
        from refresh_sentiment import get_refreshed_sentiment
        import pandas as pd
        
        # Run the refresh sentiment logic
        await asyncio.get_event_loop().run_in_executor(
            None, get_refreshed_sentiment
        )
        csv_path = "data/sector_sentiment_analysis_summary.csv"
        df = pd.read_csv(csv_path, dtype=str)
        
        if df.empty:
            return {"status": "error", "message": "Sentiment data file is empty."}
        
        sentiment_data = {}
        for _, row in df.iterrows():
            sector = row['Sector']
            latest_period = df.columns[-1]
            sentiment_value = row[latest_period]
            
            # Handle NaN/None values
            if pd.isna(sentiment_value) or sentiment_value == 'nan':
                sentiment_data[sector] = "(N/A, 0)"
            else:
                # It's already a string now, just clean it
                str_value = str(sentiment_value).strip()
                # Remove quotes if pandas added them
                if str_value.startswith('"') and str_value.endswith('"'):
                    str_value = str_value[1:-1]
                sentiment_data[sector] = str_value
        
        return {
            "status": "success",
            "data": {
                "sectors": sentiment_data,
                "period": latest_period,
                "total_sectors": len(sentiment_data)
            },
            "message": "Existing sentiment data loaded successfully"
        }
    
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}

        
        