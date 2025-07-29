# DeepGreen: RAG-Powered Financial Analyst Assistant

*A retrieval-augmented generation (RAG) chatbot for extracting insights from financial documents.*

---

## **Overview**

DeepGreen is an advanced financial analyst assistant powered by Retrieval-Augmented Generation (RAG). It enables users to quickly extract precise, sourced insights from a diverse range of financial documents and provides dynamic market outlooks based on in-depth sentiment analysis of earnings transcripts. Designed for accuracy and efficiency, DeepGreen streamlines financial research by leveraging cutting-edge AI and robust data handling.

---

## **Features**

* **Document Ingestion**: Seamlessly upload financial documents such as PDFs, spreadsheets, or plain text files for instant processing and integration into the knowledge base.
* **Precision Q&A**: Get highly accurate and sourced answers to specific questions directly from the uploaded documents (e.g., detailed figures from earnings reports, strategic information from annual reports).
* **Market Analysis**: Provides dynamic outlooks for different market sectors, powered by sophisticated sentiment analysis of company earnings transcripts.

---

## **Key Innovations & Unique Features**

DeepGreen stands out with several distinct features that enhance its reliability, data integrity, and analytical capabilities:

1.  **Dual-Layer Document Deduplication for Content Consistency:**
    DeepGreen employs a sophisticated two-stage approach to prevent redundant document ingestion, ensuring the vector database remains clean and efficient:
    * **Filename-Based Skip:** During the ingestion process, the system efficiently checks if a file (by its base name) has already been processed and logged, skipping it to avoid unnecessary reprocessing.
    * **Chunk-Content ID Check:** Beyond just filenames, the system generates unique IDs for individual document chunks. Before adding a chunk to the ChromaDB, it performs a granular check to confirm that a chunk with that exact ID does not already exist. This ensures true content-based consistency, preventing duplicates even if files are renamed or re-ingested.

2.  **High-Resolution PDF Parsing with Intelligent Table Extraction:**
    The ingestion pipeline prioritizes rich, accurate data extraction from complex financial PDFs:
    * **Advanced Parsing Strategy:** DeepGreen utilizes `Unstructured.io` with a `strategy="hi_res"` and `hi_res_model_name="yolox"`. This configuration leverages advanced computer vision models to accurately identify and extract various elements (text, tables, figures) from visually complex documents, including scanned PDFs.
    * **Direct Table Inference to Markdown:** Crucially, `infer_table_structure=True` ensures that `Unstructured.io` attempts to understand the tabular layout within PDFs. When tables are detected, `Unstructured.io` often converts them directly into **Markdown table format** within the extracted text content. This is a significant advantage, as Markdown tables are highly structured and readily interpretable by Large Language Models, enabling the LLM to accurately read and reason about financial data presented in tables. The LLM's system prompt is also explicitly designed to handle Markdown table interpretation, ensuring seamless integration.

3.  **Continuous Chat Functionality with Session-Based Context:**
    DeepGreen provides a fluid, multi-turn conversational experience:
    * **Session Cookie Management:** The API uses HTTP session cookies to uniquely identify and maintain each user's conversation.
    * **Persistent Conversation History:** A dedicated `ChatContext` object associated with each session stores the entire history of queries and responses.
    * **Contextual LLM Prompting:** This historical context is dynamically injected into the LLM's system prompt, allowing the LLM to understand follow-up questions, remember previous facts, and provide more coherent and relevant answers over multiple turns, simulating a natural dialogue.

4.  **Traceable LLM Answers with Source Attribution:**
    Ensuring transparency and trust in financial analysis is paramount:
    * **Source Prepending:** During the chunking phase of ingestion, the original filename (`[Source: filename]`) is prepended to the content of each document chunk.
    * **Automatic Source Extraction:** When the RAG system retrieves relevant chunks, these embedded source tags are automatically identified and collected.
    * **Inclusion in Response:** The extracted source names are then included in the API response returned to the user, allowing for immediate verification of the information's origin.

5.  **Market Capitalization-Weighted Sector Sentiment Analysis:**
    DeepGreen's sentiment analysis provides a more realistic and impactful view of market dynamics:
    * **Beyond Simple Averages:** Instead of a simple average, the sentiment changes of individual companies within a sector are weighted by their market capitalization.
    * **Reflecting Market Impact:** This method ensures that the sentiment shifts of larger, more influential companies have a proportionally greater impact on the overall sector sentiment score, providing more financially relevant insights that accurately reflect real-world market dynamics.

6.  **Intelligent Period Tracking for Automated Sentiment Analysis:**
    The sentiment analysis pipeline is designed for seamless, continuous operation:
    * **Automated Next Period Detection:** The system automatically reads its historical summary CSV to determine the latest analyzed period and then intelligently calculates the next chronological quarter-over-quarter period for analysis, eliminating manual date adjustments.
    * **Transcript Availability Check:** Before conducting sentiment analysis for a given company, the system rigorously verifies that valid earnings call transcripts for *both* required quarters are available (either pre-existing in the database or successfully fetched from an API). If any transcript is missing, that company is gracefully skipped for the current analysis run, ensuring the robustness and accuracy of the results.

---

## **Documentation**

For a more in-depth understanding of DeepGreen's architecture, implementation details, and advanced configurations, please refer to our comprehensive project documentation:

* **[DeepGreen Project Documentation (PDF)](./docs/Project_documentation.pdf)**
