import os
import json
import zipfile
from datetime import datetime
import shutil
import csv
import logging
from pathlib import Path
import hashlib
import time
from uuid import uuid4
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, NarrativeText, Table, Image, ListItem, Text, CompositeElement
import html2text 
import hashlib
from uuid import uuid4
# from langchain_unstructured import UnstructuredLoader # No longer needed
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain_community.document_loaders import UnstructuredPDFLoader # No longer needed
import numpy as np

from dotenv import load_dotenv
import re

load_dotenv() # Load environment variables from .env file

logger = logging.getLogger(__name__) # Get logger specific to ingest_services

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define default paths
DEFAULT_DATA_DIR = "./data"
DEFAULT_CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "Financial_Reports" # Consistent collection name

# --- Your provided classes and functions, adapted for integration ---

def convert_unstructured_elements_to_langchain_documents(unstructured_elements, combine_table_context=True):
    """
    Converts a list of unstructured elements into a list of Langchain Document objects.
    Converts HTML tables to Markdown for better embedding quality.
    Propagates useful metadata from unstructured elements.
    Correctly accesses attributes from ElementMetadata objects.

    Args:
        unstructured_elements: List of unstructured elements from partition_pdf.
        combine_table_context: If True, combines tables with the immediate preceding
                               and immediate succeeding text elements (if they are text-like).
    """
    langchain_documents = []
    i = 0 # Manual index for controlling the loop

    # Initialize html2text converter for tables
    h = html2text.HTML2Text()
    h.body_width = 0 # Disable line wrapping for tables
    h.ignore_images = True
    h.bypass_tables = False # Crucial: tells html2text to process tables

    while i < len(unstructured_elements):
        element = unstructured_elements[i]
        page_content = ""
        final_metadata = {}

        # Base metadata for the current element
        base_element_metadata = {
            "element_type": type(element).__name__,
            "element_index": i,
            "source": getattr(element.metadata, "filename", "unknown"),
            "page_number": getattr(element.metadata, "page_number", None)
        }
        final_metadata.update(base_element_metadata)

        # --- Helper function to get table content in preferred format ---
        def get_table_content_for_embedding(table_element):
            html_content = getattr(table_element.metadata, "text_as_html", None)

            if html_content:
                try:
                    # Convert HTML to Markdown using html2text
                    markdown_from_html = h.handle(html_content)
                    # Clean up extra newlines that html2text might add around tables
                    markdown_from_html = markdown_from_html.strip()
                    # A small heuristic to check if markdown conversion was meaningful for a table
                    if len(markdown_from_html) > 5 and ('|' in markdown_from_html or '-' in markdown_from_html): # Basic check for table-like structure
                        return markdown_from_html, "markdown"
                    else:
                        # If html2text yields very little or non-table-like markdown, fall back to plain text
                        return getattr(table_element, "text", "").strip(), "plain_text_from_html_fallback"
                except Exception as e:
                    # Fallback if html2text fails for some reason
                    print(f"Warning: html2text conversion failed for table {table_element.metadata.page_number} (index {i}): {e}. Falling back to plain text.")
                    return getattr(table_element, "text", "").strip(), "plain_text_conversion_error"
            else:
                return getattr(table_element, "text", "").strip(), "plain_text"


        # --- Check for table with context combining ---
        if isinstance(element, Table) and combine_table_context:
            content_parts = []

            table_raw_content, table_format = get_table_content_for_embedding(element)
            final_metadata["table_format"] = table_format
            table_content_with_prefix = "Table:\n" + table_raw_content

            # --- Get content from the immediate preceding element (if it's text-like) ---
            prev_content = None
            if i > 0:
                prev_element = unstructured_elements[i - 1]
                if isinstance(prev_element, (Title, NarrativeText, ListItem, Text, CompositeElement)):
                    if prev_element.text and prev_element.text.strip():
                        prev_content = prev_element.text.strip()
                        if prev_element.metadata.page_number is not None and \
                           prev_element.metadata.page_number < final_metadata.get("page_number", float('inf')):
                            final_metadata["page_number_start"] = prev_element.metadata.page_number

            if prev_content:
                content_parts.append(prev_content)

            content_parts.append(table_content_with_prefix)

            # --- Get content from the immediate succeeding element (if it's text-like) ---
            next_content = None
            elements_to_advance_past = 1 # Start with the current table itself
            if i + 1 < len(unstructured_elements):
                next_element = unstructured_elements[i + 1]
                if isinstance(next_element, (Title, NarrativeText, ListItem, Text, CompositeElement)):
                    if next_element.text and next_element.text.strip():
                        next_content = next_element.text.strip()
                        if next_element.metadata.page_number is not None and \
                           next_element.metadata.page_number > final_metadata.get("page_number", 0):
                            final_metadata["page_number_end"] = next_element.metadata.page_number
                        elements_to_advance_past += 1 # We've consumed this next element as well

            if next_content:
                content_parts.append(next_content)

            page_content = "\n\n".join(filter(None, content_parts))
            final_metadata["element_type"] = "TableWithContext"
            final_metadata["original_element_type"] = "Table"

            if "page_number_start" not in final_metadata and final_metadata["page_number"] is not None:
                final_metadata["page_number_start"] = final_metadata["page_number"]
            if "page_number_end" not in final_metadata and final_metadata["page_number"] is not None:
                final_metadata["page_number_end"] = final_metadata["page_number"]

            i += elements_to_advance_past
            # No continue here, as we want to create a document for this combined content

        # --- Handle other element types (text-based, images) or tables when context combining is off ---
        else:
            # THIS IS THE KEY LOGIC TO PREVENT DUPLICATION OF THE PREVIOUS ELEMENT:
            is_text_and_will_be_prev_context = False
            if combine_table_context and i + 1 < len(unstructured_elements):
                next_element_candidate = unstructured_elements[i + 1]
                if isinstance(next_element_candidate, Table) and \
                   isinstance(element, (Title, NarrativeText, ListItem, Text, CompositeElement)):
                    is_text_and_will_be_prev_context = True

            if is_text_and_will_be_prev_context:
                i += 1
                continue # Skip this element as it will be included with the next table

            # Now, process the current element as a standalone document
            if isinstance(element, Table): # Handles standalone tables (combine_table_context is False)
                table_raw_content, table_format = get_table_content_for_embedding(element)
                page_content = "Table:\n" + table_raw_content
                final_metadata["table_format"] = table_format
                final_metadata["element_type"] = "Table"

            elif hasattr(element, 'text') and element.text:
                page_content = element.text.strip()

            elif isinstance(element, Image):
                image_filename = getattr(element.metadata, "filename", "No filename available")
                image_page_number = getattr(element.metadata, "page_number", "unknown")
                page_content = f"[Image on page {image_page_number}: {image_filename}]"
                final_metadata["image_info"] = True

            else: # If an element type is not handled, skip it
                i += 1
                continue # Skip to the next element if not handled

            i += 1 # Advance the index for standalone elements

        # --- Common metadata propagation for all document types ---
        coordinates = getattr(element.metadata, "coordinates", None)
        if coordinates: final_metadata["coordinates"] = str(coordinates)

        # Filter out None values in final_metadata for cleaner Langchain Documents
        final_metadata = {k: v for k, v in final_metadata.items() if v is not None}

        # Create Langchain Document only if there's actual content
        if page_content:
            langchain_doc = Document(page_content=page_content, metadata=final_metadata)
            langchain_documents.append(langchain_doc)

    return langchain_documents

def generate_document_id(chunk: Document, chunk_index: int) -> Document:
    """
    Generates a unique and consistent ID for a document chunk and adds it to metadata.

    The ID is composed of:
    - Base filename (e.g., "iFASTCorp-AR2023.pdf")
    - Zero-padded chunk index (e.g., "0001")
    - A truncated hash of the chunk's content for content-based consistency.

    Args:
        chunk (Document): The LangChain Document object for the chunk.
        chunk_index (int): The numerical index of the chunk within its parent document/file.
                             This is crucial for uniqueness when content might be repeated.

    Returns:
        Document: The modified LangChain Document object with the 'id' in its metadata.
    """
    if chunk_index is None:
        raise ValueError("chunk_index must be provided for generating chunk IDs.")

    # Get a stable filename from the source metadata
    source = chunk.metadata.get('source', 'unknown')
    source_name = os.path.basename(source) if source != 'unknown' else 'unknown'

    # Generate a hash of the chunk's content
    normalized_content = " ".join(chunk.page_content.split())
    content_hash = hashlib.md5(normalized_content.encode('utf-8')).hexdigest()

    # Combine elements to form a unique ID
    chunk_id = f"{source_name}_{chunk_index:05d}_{content_hash[:10]}"
    chunk.metadata["id"] = chunk_id

    return chunk

def add_documents_in_batches(vector_store, documents, ids, batch_size=2000):
    """
    Add documents to the vector store in batches to avoid processing limits.

    Args:
        vector_store: ChromaDB vector store instance
        documents: List of Document objects to add
        ids: List of corresponding document IDs
        batch_size: Number of documents to add per batch (default: 2000)
    """
    if len(documents) != len(ids):
        raise ValueError("Number of documents must match number of IDs")

    total_docs = len(documents)
    if total_docs == 0:
        print("No documents to add.")
        return

    print(f"Adding {total_docs} documents in batches of {batch_size}...")

    for i in range(0, total_docs, batch_size):
        batch_end = min(i + batch_size, total_docs)
        batch_docs = documents[i:batch_end]
        batch_ids = ids[i:batch_end]

        batch_num = (i // batch_size) + 1
        total_batches = (total_docs + batch_size - 1) // batch_size

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_docs)} documents)...")

        try:
            vector_store.add_documents(batch_docs, ids=batch_ids)
            print(f"✅ Successfully added batch {batch_num}/{total_batches}")
        except Exception as e:
            print(f"❌ Error adding batch {batch_num}/{total_batches}: {e}")
            continue

    print(f"Completed batch addition of {total_docs} documents.")

def filter_complex_metadata(chunks: list[Document]) -> list[Document]:
    """
    Filters out complex metadata types that ChromaDB might not handle well,
    keeping only simple types (str, int, float, bool, list of simple types).
    Specifically handles the 'coordinates' field which can be a complex object.
    """
    filtered_chunks = []
    for chunk in chunks:
        new_metadata = {}
        for key, value in chunk.metadata.items():
            if key == 'coordinates':
                # Convert coordinates to a string representation if it's a complex object
                if hasattr(value, '__dict__'): # Check if it's an object
                    new_metadata[key] = str(value)
                else: # Assume it's already a simple type or list/tuple
                    new_metadata[key] = value
            elif isinstance(value, (str, int, float, bool)):
                new_metadata[key] = value
            elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
                new_metadata[key] = value
            else:
                # Optionally, log or convert other complex types if needed
                logging.debug(f"Skipping complex metadata key '{key}' with type {type(value)} for chunk.")
        filtered_chunks.append(Document(page_content=chunk.page_content, metadata=new_metadata))
    return filtered_chunks

# --- Main ingestion function, adapted from your provided main() ---
def ingest_documents_to_chroma(
    global_vector_store: Chroma, # <--- NEW REQUIRED ARGUMENT
    global_embeddings: OllamaEmbeddings, # <--- NEW REQUIRED ARGUMENT
    data_dir: str = DEFAULT_DATA_DIR,
    use_adobe_api: bool = False,
    reset_collection: bool = False # This parameter might become less relevant if always using global
):
    """
    Processes documents from the specified data_dir and ingests them into ChromaDB.
    This function will add new documents to an existing collection or create a new one.
    It's designed to be called multiple times.
    """
    logger.info(f"Starting ingestion process from {data_dir}")

    if not os.path.exists(data_dir):
        logger.warning(f"Warning: Data directory {data_dir} does not exist. No documents to load.")
        return

    def get_pdf_file_paths(data_path):
        """
        Most efficient method to get PDF file paths
        Uses os.scandir() - fastest and most memory efficient for file discovery
        """
        try:
            with os.scandir(data_path) as entries:
                return [os.path.join(data_path, entry.name)
                        for entry in entries
                        if entry.is_file() and entry.name.lower().endswith('.pdf')]
        except FileNotFoundError:
            logger.error(f"Directory not found: {data_path}")
            return []

    file_paths = get_pdf_file_paths(data_dir)

    if not file_paths:
        logger.info("No PDF files found to process. Exiting ingestion.")
        return
    else:
        logger.info(f"Processing {len(file_paths)} PDF files...")

    embeddings_for_ingestion = global_embeddings # Use the passed global embeddings
    vector_store_for_ingestion = global_vector_store 

    #if reset_collection:
        #vector_store.reset_collection()
        #logger.info("✅ Collection has been reset.")

    log_dir = "/app/logs"
    added_files_path = os.path.join(log_dir, "Added_files.txt")

    added_files = set()
    try:
        with open(added_files_path, "r") as f:
            added_files = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded {len(added_files)} previously added files from Added_files.txt")
    except FileNotFoundError:
        logger.info("No previous additions found. Starting fresh.")

    newly_added_files = set() # To track files added in this run

    for i, file_path in enumerate(file_paths):
        file_basename = os.path.basename(file_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING FILE {i+1}/{len(file_paths)}: {file_basename}")
        logger.info(f"Inside loop: vector_store_for_ingestion ID: {id(vector_store_for_ingestion)}")
        logger.info(f"{'='*60}")

        if file_basename in added_files:
            logger.info(f"File {file_basename} was previously added. Skipping.")
            continue

        langchain_docs = []
        if use_adobe_api:
            logger.warning(f"Adobe API extraction failed. Falling back to Unstructured.io.")
        else:
            logger.info(f"Using Unstructured.io to partition {file_basename}...")

        if not langchain_docs: # If Adobe API was not used, or failed, use Unstructured
            try:
                # --- ADDED LOGGING AROUND partition_pdf ---
                logger.info(f"Attempting to partition {file_basename} with Unstructured.io (strategy='hi_res', model='yolox'). This may take time...")
                elements = partition_pdf(
                    filename=file_path,
                    strategy="hi_res",
                    infer_table_structure=True,
                    hi_res_model_name="yolox",
                    chunking_strategy="basic",
                    # Consider adding infer_ocr_text=False here if your PDFs are text-based and not scanned.
                    # This can drastically reduce memory and time.
                    # infer_ocr_text=False,
                )
                logger.info(f"SUCCESS: Finished partitioning {file_basename}. Found {len(elements)} unstructured elements.")

                logger.info("Converting unstructured elements to Langchain Documents...")
                langchain_docs = convert_unstructured_elements_to_langchain_documents(elements)
                logger.info(f"SUCCESS: Converted to {len(langchain_docs)} Langchain Documents.")
            except Exception as e:
                logger.error(f"Error partitioning or converting {file_basename} with Unstructured: {e}", exc_info=True)
                logger.error(f"Skipping {file_basename} due to processing error.")
                continue

        if not langchain_docs:
            logger.warning(f"No Langchain documents were created from {file_basename}. Skipping.")
            continue

        # Display sample content
        if langchain_docs:
            logger.info("\n--- Content of the first converted Langchain document (first 500 chars) ---")
            logger.info(f"{langchain_docs[0].page_content[:500]}")
            logger.info("----------------------------------------------------------\n")
            logger.info(f"Length of text in the first document: {len(langchain_docs[0].page_content)}")

        # Initialize text splitter
        logger.info("Initializing RecursiveCharacterTextSplitter for chunking...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        chunks = text_splitter.split_documents(langchain_docs)
        logger.info(f"Original documents split into {len(chunks)} chunks.")

        # Filter complex metadata to avoid Chroma issues
        logger.info("Filtering complex metadata...")
        filtered_chunks = filter_complex_metadata(chunks)
        logger.info(f"Filtered chunks to {len(filtered_chunks)} valid documents.")

        # Display sample chunks
        logger.info("\n--- Example of the first 3 final chunks ---")
        for j, chunk in enumerate(filtered_chunks[:3]):
            logger.info(f"Chunk {j+1} (Length: {len(chunk.page_content)}):")
            logger.info(chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content)
            logger.info(f"Metadata: {chunk.metadata}")
            logger.info("-" * 80)

        try:
            existing_ids_in_db = set(vector_store_for_ingestion.get(include=[])['ids'])
            logger.info(f"Number of existing documents in DB before adding current file's chunks: {len(existing_ids_in_db)}")
        except Exception as e:
            existing_ids_in_db = set()
            logger.warning(f"Could not retrieve existing document IDs (likely collection not yet created or error): {e}")
            logger.info("Proceeding assuming collection might be empty or new.")

        chunks_to_add = []
        ids_to_add = []

        for j, chunk in enumerate(filtered_chunks):
            # Prepend filename to the chunk content
            chunk.page_content = f"[Source: {file_basename}]\n\n{chunk.page_content}"

            # Generate ID and add to chunk metadata
            chunk_with_id = generate_document_id(chunk, j)
            chunk_id = chunk_with_id.metadata["id"]

            if chunk_id not in existing_ids_in_db:
                chunks_to_add.append(chunk_with_id)
                ids_to_add.append(chunk_id)
            else:
                logger.info(f"Chunk with ID {chunk_id} already exists in DB. Skipping.")

        if chunks_to_add:
            logger.info(f"👉 Adding {len(chunks_to_add)} new chunks from {file_basename} to ChromaDB.")
            # --- ADDED LOGGING AROUND ChromaDB ADD ---
            logger.info(f"Attempting to add {len(chunks_to_add)} documents to ChromaDB for {file_basename}...")
            add_documents_in_batches(vector_store_for_ingestion, chunks_to_add, ids_to_add)
            logger.info(f"SUCCESS: Added {len(chunks_to_add)} documents to ChromaDB for {file_basename}.")
            # --- END ADDED LOGGING ---
            newly_added_files.add(file_basename)
        else:
            logger.info(f"No new chunks to add for {file_basename} after deduplication.")

    # Update Added_files.txt with all files successfully processed in this run
    if newly_added_files:
        logger.info(f"\nUpdating Added_files.txt with {len(newly_added_files)} newly processed files...")
        with open(added_files_path, "a") as f:
            for filename in newly_added_files:
                f.write(f"{filename}\n")
        logger.info("Added_files.txt updated.")
    else:
        logger.info("No new files were added during this ingestion run.")

    logger.info("\nIngestion process complete for all specified files.")

# This block allows the script to be run directly (e.g., by run.sh for initial fill, or via docker exec)
if __name__ == "__main__":
    # Example of how to run it directly
    # You can modify these parameters or pass them from a shell script
    # For initial setup, you might want to reset the collection:
    # ingest_documents_to_chroma(reset_collection=True, use_adobe_api=False)
    # To use Adobe API (if configured):
    # ingest_documents_to_chroma(use_adobe_api=True)
    ingest_documents_to_chroma(use_adobe_api=False, reset_collection=False) # Default behavior