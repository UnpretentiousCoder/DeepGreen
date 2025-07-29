import hashlib
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
import numpy as np
import requests
import json
import os
import pandas as pd
from collections import defaultdict
import time # For adding a small delay between API calls if needed
import yfinance as yf

# --- API Configuration ---
API_key = "kxWxAMwqFpkYdi1DlD5OyA==uAGXDvBolSXQKC8N"

# --- ChromaDB Persistence Directory ---
TRANSCRIPT_CHROMA_PATH = "./ç"
os.makedirs(TRANSCRIPT_CHROMA_PATH, exist_ok=True)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

load_dotenv()

def generate_document_id(chunk, chunk_index):
    """
    Generates a unique and consistent ID for a document chunk.
    
    The ID is composed of:
    - Source name (from metadata)
    - Zero-padded chunk index
    - A truncated hash of the chunk's content for content-based consistency.
    
    Args:
        chunk (Document): The LangChain Document object for the chunk.
        chunk_index (int): The numerical index of the chunk within its parent document/file.
                            This is crucial for uniqueness when content might be repeated.
        
    Returns:
        Document: The modified LangChain Document object with the 'id' added to its metadata.
    """
    if chunk_index is None:
        raise ValueError("chunk_index must be provided for generating chunk IDs.")

    source = chunk.metadata.get('source', 'unknown')
    
    normalized_content = " ".join(chunk.page_content.split())
    content_hash = hashlib.md5(normalized_content.encode('utf-8')).hexdigest()
    
    chunk_id = f"{source}_{chunk_index:05d}_{content_hash[:10]}"
    chunk.metadata["id"] = chunk_id

    return chunk

def get_tickers_and_sectors_from_stockanalysis():
    """
    Fetches ticker symbols and their sectors from the StockAnalysis.com API.

    Returns:
        pd.DataFrame: A DataFrame with 'Symbol' and 'StockAnalysis Sector' columns,
                      or an empty DataFrame if retrieval fails.
    """
    # This is the API URL identified from your network inspection
    api_url = "https://stockanalysis.com/api/screener/s/d/sector.json"
    
    try:
        print(f"Fetching tickers and sectors from StockAnalysis.com API: {api_url}")
        response = requests.get(api_url)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        json_data = response.json()
        
        # Check the JSON structure based on your screenshot
        if json_data.get('status') == 200 and 'data' in json_data and 'data' in json_data['data']:
            extracted_data = []
            # Iterate through the inner 'data' list, which contains [Ticker, Sector] lists
            for item in json_data['data']['data']:
                if len(item) >= 2: # Ensure the list has at least two elements (ticker and sector)
                    ticker = item[0].strip()
                    sector = item[1].strip()
                    extracted_data.append({'Symbol': ticker, 'GICS_Sector': sector})
            
            df = pd.DataFrame(extracted_data)
            # Remove any duplicate tickers and sort by symbol
            df = df.drop_duplicates(subset=['Symbol']).sort_values('Symbol').reset_index(drop=True)
            print(f"Found {len(df)} tickers with sectors from StockAnalysis.com.")
            return df
        else:
            print(f"Error: Unexpected JSON structure or status from StockAnalysis.com API. Response: {json_data}")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from StockAnalysis.com API: {e}")
        return pd.DataFrame()
    except ValueError as e: # For JSON decoding errors if the response isn't valid JSON
        print(f"Error decoding JSON response from StockAnalysis.com API: {e}")
        return pd.DataFrame()
    
def get_company_market_cap(ticker_symbol):
    """
    Fetches market capitalization for a given company ticker symbol using yfinance.

    Args:
        ticker_symbol (str): The stock ticker symbol.

    Returns:
        float or None: The market capitalization, or None if the data cannot be retrieved.
    """
    try:
        company = yf.Ticker(ticker_symbol)
        info = company.info
        market_cap = info.get('marketCap')
        return market_cap
    except Exception as e:
        print(f"Error fetching market cap for {ticker_symbol}: {e}")
        return None

def fetch_transcript_from_api(ticker, year, quarter, api_key):
    """
    Fetches the earnings transcript from API-Ninjas, handles API response,
    and returns the extracted transcript content string if available and valid, otherwise None.
    """
    url = 'https://api.api-ninjas.com/v1/earningstranscript?ticker={}&year={}&quarter={}'.format(ticker, year, quarter)
    headers = {'X-Api-Key': api_key}
    
    try:
        response = requests.get(url, headers=headers, timeout=15) # Increased timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        
        api_data = response.json() # Parse JSON response
        
        parsed_transcript = None
        # API-Ninjas typically returns a list of dicts, even if only one result
        if isinstance(api_data, list) and len(api_data) > 0 and 'transcript' in api_data[0]:
            parsed_transcript = api_data[0]['transcript']
        # Fallback for single dict response (less common for this API, but good to have)
        elif isinstance(api_data, dict) and 'transcript' in api_data:
            parsed_transcript = api_data['transcript']
        
        if parsed_transcript:
            return parsed_transcript
        else:
            return None
            
    except requests.exceptions.HTTPError as http_err:
        return None
    except requests.exceptions.ConnectionError as conn_err:
        return None
    except requests.exceptions.Timeout as timeout_err:
        return None
    except requests.exceptions.RequestException as req_err:
        return None
    except json.JSONDecodeError:
        return None
    except Exception as e:
        return None

def add_documents_in_batches(vector_store, documents, ids, batch_size=2000):
    """
    Add documents to the vector store in batches to avoid processing limits.
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

def get_companies_by_sector_from_csv(csv_path, num_per_sector, api_key, first_year, first_quarter, second_year, second_quarter):
    """
    Reads a CSV, groups companies by sector, and selects a specified number of tickers per sector
    that have transcripts for BOTH specified quarters.
    Returns a DataFrame containing 'Symbol', 'GICS_Sector', and 'Market_Cap' for the selected companies(has transcripts).

    Args:
        csv_path (str): Path to the CSV file containing 'Symbol', 'GICS_Sector', and 'Market_Cap'.
        num_per_sector (int): Desired number of *valid* tickers to select from each sector.
        api_key (str): Your API-Ninjas API key.
        first_year (int), first_quarter (int): Year and quarter for the first transcript.
        second_year (int), second_quarter (int): Year and quarter for the second transcript.

    Returns:
        pd.DataFrame: A DataFrame with 'Symbol', 'GICS_Sector', and 'Market_Cap' columns for the selected companies.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found for ticker selection.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file '{csv_path}': {e}")
        return pd.DataFrame()

    # Ensure all required columns exist, including 'Market_Cap'
    required_columns = ['Symbol', 'GICS_Sector', 'MarketCap']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV file '{csv_path}' must contain all of {required_columns} columns.")
        return pd.DataFrame()
    
    # Convert Market_Cap to numeric, handling errors
    df['MarketCap'] = pd.to_numeric(df['MarketCap'], errors='coerce')
    df.dropna(subset=['MarketCap'], inplace=True) # Drop rows where Market_Cap couldn't be converted

    selected_companies_df = pd.DataFrame(columns=required_columns) # Update columns here
    
    print(f"\nAttempting to select {num_per_sector} tickers with transcripts for {first_year} Q{first_quarter} and {second_year} Q{second_quarter} from each GICS Sector from '{csv_path}'...")
    
    # Iterate through each unique sector
    for sector, group in df.groupby('GICS_Sector'):
        print(f"\n--- Processing Sector: {sector} ---")
        selected_count = 0
        tickers_for_this_sector = []
        
        # Sort group by Symbol (or Market_Cap, if a preference)
        group_sorted = group.sort_values(by='Symbol').reset_index(drop=True)

        for _, row in group_sorted.iterrows():
            ticker = row['Symbol']
            market_cap = row['MarketCap'] # Get market cap here
            
            # Skip if market cap is 0 or NaN, as it would cause issues with weighting
            if pd.isna(market_cap) or market_cap <= 0:
                print(f"  Skipping {ticker} (invalid Market Cap: {market_cap}).{' ' * 30}", end='\r')
                continue

            if selected_count >= num_per_sector:
                break # Already found enough for this sector
            
            # Print on the same line to avoid spamming console, indicating progress
            print(f"  Checking {ticker} for transcripts... Current selected for {sector}: {selected_count}/{num_per_sector}", end='\r')
            
            # Check for transcript availability for both quarters using the enhanced function
            transcript_q1_content = fetch_transcript_from_api(ticker, first_year, first_quarter, api_key)
            time.sleep(0.1) # Small delay to respect API rate limits
            transcript_q2_content = fetch_transcript_from_api(ticker, second_year, second_quarter, api_key)
            time.sleep(0.1)

            if transcript_q1_content and transcript_q2_content:
                tickers_for_this_sector.append({'Symbol': ticker, 'GICS_Sector': sector, 'MarketCap': market_cap}) # Include Market_Cap
                selected_count += 1
                # Overwrite the previous line with a success message
                print(f"  ✅ Found transcripts for {ticker}. Selected {selected_count}/{num_per_sector} for {sector}.{' ' * 30}") # Add padding to clear line
            else:
                # Clear the line if skipping to avoid leftover messages from previous `end='\r'`
                print(f"  ❌ Skipping {ticker} (no transcripts for Q{first_quarter} {first_year} or Q{second_quarter} {second_year}).{' ' * 30}") 

        if tickers_for_this_sector:
            sector_df = pd.DataFrame(tickers_for_this_sector)
            if not sector_df.empty:
                selected_companies_df = pd.concat([selected_companies_df, sector_df], ignore_index=True)
            print(f"--- Finished selecting for {sector}. Total selected: {len(tickers_for_this_sector)} ---")
        else:
            print(f"--- No companies with available transcripts found for {sector} ---")

    print(f"\nTotal unique tickers selected for analysis across all sectors: {len(selected_companies_df)}")
    return selected_companies_df

def group_companies_by_gics_sector(input_csv_path, output_csv_path):
    """
    Loads a CSV, groups companies by 11 specified GICS sectors,
    and saves the grouped data to a new CSV.

    Args:
        input_csv_path (str): Path to the input CSV file (e.g., 'combined_stock_data.csv').
        output_csv_path (str): Path where the grouped output CSV will be saved.
    """
    # Define the 11 target GICS sectors
    target_gics_sectors = [
        "Communication Services",
        "Consumer Discretionary",
        "Consumer Staples",
        "Energy",
        "Financials",
        "Healthcare",
        "Industrials",
        "Technology",
        "Materials",
        "Real Estate",
        "Utilities"
    ]

    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file '{input_csv_path}' not found.")
        return

    print(f"Loading data from '{input_csv_path}'...")
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Ensure the 'GICS_Sector' column exists
    if 'GICS_Sector' not in df.columns:
        print(f"Error: 'GICS_Sector' column not found in '{input_csv_path}'.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Convert GICS_Sector to string type and handle potential NaN values
    df['GICS_Sector'] = df['GICS_Sector'].astype(str).fillna('Unknown')

    # Filter out companies that are not in the 11 target sectors
    initial_count = len(df)
    df_filtered = df[df['GICS_Sector'].isin(target_gics_sectors)].copy()
    filtered_count = len(df_filtered)

    if initial_count != filtered_count:
        print(f"\nFiltered out {initial_count - filtered_count} companies not belonging to the 11 target GICS sectors.")
        # Optionally, print which sectors were filtered out
        non_target_sectors = df[~df['GICS_Sector'].isin(target_gics_sectors)]['GICS_Sector'].unique().tolist()
        if non_target_sectors:
            print(f"Non-target sectors found: {non_target_sectors}")
    
    if df_filtered.empty:
        print("No companies found matching the specified GICS sectors after filtering. Exiting.")
        return

    # Sort the DataFrame by GICS_Sector and then by Symbol for clear grouping in the output
    df_grouped_sorted = df_filtered.sort_values(by=['GICS_Sector', 'Symbol']).reset_index(drop=True)

    print("\n--- Summary of Companies per GICS Sector ---")
    sector_counts = df_grouped_sorted['GICS_Sector'].value_counts().sort_index()
    print(sector_counts.to_string())

    print(f"\nSaving grouped data to '{output_csv_path}'...")
    try:
        df_grouped_sorted.to_csv(output_csv_path, index=False)
        print("Successfully saved grouped data.")
        print(f"\nFirst 10 rows of the grouped data in '{output_csv_path}':")
        print(df_grouped_sorted.head(10).to_string())
    except Exception as e:
        print(f"Error saving grouped data to CSV: {e}")

def get_sentiment(): #THIS FETCHES THE LASTEST DATA
    COLLECTION_NAME = "Financial_Reports"  # Consistent collection name
    CSV_FILE = "data/sector_sentiment_analysis_summary.csv"
    
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE):
        # Default periods if CSV doesn't exist
        first_transcript_year = 2025
        first_transcript_quarter = 1
        second_transcript_year = 2025
        second_transcript_quarter = 2
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
                
                print(f"\nFound existing analysis. Automatically selecting next period:")
                print(f"First transcript: Q{first_transcript_quarter} {first_transcript_year}")
                print(f"Second transcript: Q{second_transcript_quarter} {second_transcript_year}")
                
            else:
                # Fallback to default if no periods found in CSV
                print(f"\nCSV file exists but no period information found. Using default periods:")
                first_transcript_year = 2024
                first_transcript_quarter = 3
                second_transcript_year = 2024
                second_transcript_quarter = 4
                print(f"First transcript: Q{first_transcript_quarter} {first_transcript_year}")
                print(f"Second transcript: Q{second_transcript_quarter} {second_transcript_year}")
                
        except Exception as e:
            print(f"\nError reading CSV file: {e}")
            print("Using default periods:")
            first_transcript_year = 2024
            first_transcript_quarter = 4
            second_transcript_year = 2025
            second_transcript_quarter = 1
            print(f"First transcript: Q{first_transcript_quarter} {first_transcript_year}")
            print(f"Second transcript: Q{second_transcript_quarter} {second_transcript_year}")
    

    if (second_transcript_year < first_transcript_year) or \
        (second_transcript_year == first_transcript_year and second_transcript_quarter <= first_transcript_quarter):
        print("\nWarning: The second transcript period should ideally be chronologically after the first.")
        print("Please ensure your chosen periods make sense for comparative analysis.")

    csv_for_tickers = 'grouped_by_gics_sector.csv' 
    num_companies_per_sector = 5 # You can adjust this number. For now, we'll aim for this many.

    print("Fetching fresh ticker and market cap data...")
    stockanalysis_df = get_tickers_and_sectors_from_stockanalysis() # Assuming this function is defined elsewhere
    
    if stockanalysis_df.empty:
        print("No data retrieved from StockAnalysis.com API. Exiting.")
        return 
    
    print(f"Retrieved {len(stockanalysis_df)} tickers from StockAnalysis.com")
    
    yfinance_market_cap_data = []
    
    # Define retry parameters for market cap fetching
    max_market_cap_retries = 3
    initial_retry_delay_seconds = 5 # Initial delay between retries

    print("Fetching Market Cap data using yfinance (with retries for failures)...")
    
    for i, row in stockanalysis_df.iterrows(): 
        ticker = row['Symbol']
        current_market_cap = np.nan # Initialize to NaN, will be updated on success
        
        # --- Immediate Retry Logic for Market Cap Fetching ---
        for attempt in range(max_market_cap_retries):
            print(f"[{i+1}/{len(stockanalysis_df)}] Fetching Market Cap for {ticker} (Attempt {attempt + 1}/{max_market_cap_retries})...")
            
            fetched_cap = get_company_market_cap(ticker) # This returns float or np.nan
            
            if pd.notna(fetched_cap): # pd.notna() checks if a value is NOT NaN (includes numbers, None, etc.)
                current_market_cap = fetched_cap
                print(f"Successfully fetched market cap for {ticker} on attempt {attempt + 1}.")
                break # Market cap obtained, exit retry loop
            else:
                # If fetched_cap is np.nan, it means get_company_market_cap failed
                print(f"Failed to fetch market cap for {ticker} on attempt {attempt + 1}. Error details should be above.")
                if attempt < max_market_cap_retries - 1:
                    # Exponential backoff for subsequent retries
                    delay = initial_retry_delay_seconds * (2 ** attempt) 
                    print(f"Retrying {ticker} after {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Max retries ({max_market_cap_retries}) exhausted for {ticker}. Market cap will remain NaN for this run.")
        # --- End Immediate Retry Logic ---

        yfinance_market_cap_data.append({'Symbol': ticker, 'MarketCap': current_market_cap})
        
        # Small general delay between tickers to be courteous to the API,
        # separate from retry delays which are for specific failures.
        time.sleep(0.1) 
    
    # Step 3: Create DataFrame and merge
    df_market_cap = pd.DataFrame(yfinance_market_cap_data)
    
    df_combined = pd.merge(
        stockanalysis_df, 
        df_market_cap, 
        on='Symbol', 
        how='left' 
    )
    
    # Ensure MarketCap column is numeric, coercing any non-numeric (like if an error slipped through) to NaN
    df_combined['MarketCap'] = pd.to_numeric(df_combined['MarketCap'], errors='coerce')
    
    # --- NEW LOGIC: Filter out rows where MarketCap is NaN ---
    initial_row_count = len(df_combined)
    df_combined = df_combined.dropna(subset=['MarketCap'])
    rows_removed_count = initial_row_count - len(df_combined)

    if rows_removed_count > 0:
        print(f"Removed {rows_removed_count} tickers from 'combined_stock_data.csv' because their market cap could not be fetched after retries.")
    else:
        print("All tickers retained in 'combined_stock_data.csv' as market caps were successfully fetched for all.")
    # --- END NEW LOGIC ---

    # Step 4: Save combined data
    combined_output_file = 'combined_stock_data.csv'
    df_combined.to_csv(combined_output_file, index=False)
    print(f"Combined data saved to '{combined_output_file}'")
    
    # Step 5: Group by GICS sector and create the final CSV
    print("Grouping companies by GICS sector...")
    # This function should likely use the df_combined for its input data, not just the raw csv_for_tickers
    # Ensure group_companies_by_gics_sector handles the MarketCap column being present.
    group_companies_by_gics_sector(combined_output_file, csv_for_tickers) # Passing combined_output_file
    print(f"Grouped data saved to '{csv_for_tickers}'")
    # Call the modified function to get pre-screened companies, now with Market_Cap

    companies_df_for_analysis = get_companies_by_sector_from_csv(
        csv_for_tickers, num_companies_per_sector, API_key, 
        first_transcript_year, first_transcript_quarter, 
        second_transcript_year, second_transcript_quarter
    )

    if companies_df_for_analysis.empty:
        print("No company tickers found for analysis with available transcripts for the specified periods. Exiting.")
        return

    # Initialize Embedding Model
    embeddings_model = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3") 
    )
    print("Ollama Embeddings Initialized.")

    # Initialize Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    # Initialize LLM (keeping your chosen model and settings)
    llm = ChatOllama(
        model="gemma3:27b",
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_ctx=16384
    )

    # Define the system prompt content for the single LLM call
    system_prompt_content = """
    You are a highly skilled quantitative financial analyst AI assistant, specializing in deep sentiment extraction and numerical scoring from company earnings call transcripts. Your analysis must be objective, data-driven, and strictly based on the provided text.

    Your responsibilities are:
    1. **Individual Quarter Analysis:** You will receive two distinct quarterly earnings call transcripts. Your first and primary step is to **analyze each quarter's transcript independently and thoroughly**, treating them as separate reporting periods.
    2. Sentiment Scoring (Per Quarter, 0-100): For each transcript, analyze the sentiment and assign a numerical Sentiment Score between 0 and 100. The score should reflect the overall tone, outlook, and key developments presented. A score of 50 denotes a truly neutral quarter. Scores below 50 indicate negative sentiment and scores above 50 indicate positive sentiment. To avoid arbitrariness, use these fixed tiers as your guide:
        •   0-20: very negative outlook
        •   21-40: negative outlook
        •   41-60: neutral outlook
        •   61-80: positive outlook
        •   81-100: very positive outlook
    3. **Comparative Analysis & Percentage Change Calculation:** After analyzing both quarters individually and assigning their respective 0-100 sentiment scores, you will compare them. Your comparison MUST include a precise **percentage change** in sentiment score from the first quarter's (Q4 of Year1) score to the second quarter's (Q1 of Year2) score, rounded to one decimal place.

    ### FORMATTING REQUIREMENTS:

    For each quarter, you MUST follow this strict format. Ensure all sections are present and accurately filled:

    Sentiment: [Positive / Neutral / Negative]
    Key Quotes: "Quote 1", "Quote 2", ...
    Analysis: [Detailed explanation of tone, growth, risks, outlook for THIS quarter]
    Sentiment Score: XX/100
    Sentiment Justification: [A concise, specific reason why THIS score was given, linking directly to transcript content.]

    After both quarters are analyzed, you MUST include this summary EXACTLY as follows:

    ---
    Percentage Change in Sentiment Score: [e.g., +15.0% or -8.5% or 0.0%]
    **(Calculation: $\left( \frac{\text{[Q1 of Year2 Sentiment Score]} - \text{[Q4 of Year1 Sentiment Score]}}{\text{[Q4 of Year1 Sentiment Score]}} \right) \times 100\%$)**
    **IMPORTANT:** In the calculation above, you MUST replace `[Q1 of Year2 Sentiment Score]` and `[Q4 of Year1 Sentiment Score]` with the *actual numerical sentiment scores* you assigned to those quarters. For example, if Q4 Year1 score was 75 and Q1 Year2 score was 80, the calculation would be: $\left( \frac{80 - 75}{75} \right) \times 100\%$
    **IMPORTANT:** If the [Q4 of Year1 Sentiment Score] (the denominator) is 0, the percentage change is undefined. In this specific case, output "N/A (Q4 of Year1 Sentiment Score was 0)" instead of a percentage.
    Change Summary: [A brief, clear explanation of what changed or trended between the two quarters' sentiments and outlooks, based on your scores and analysis.]

    ### SCORING GUIDELINES (USE THESE TO DETERMINE THE SPECIFIC NUMBER):
    - **90-100 (Strongly Positive):** Overwhelmingly bullish, significant outperformance, highly optimistic guidance, clear competitive advantages, strong growth drivers. Assign higher scores (95-100) for truly exceptional results.
    - **70-89 (Positive):** Generally positive, met or slightly exceeded expectations, positive outlook with minor manageable concerns, steady growth.
    - **50-69 (Neutral/Mixed):** In-line performance, balanced outlook, some positive and some negative factors, no strong directional bias. Scores closer to 50 are more neutral, higher in this range lean slightly positive, lower slightly negative.
    - **30-49 (Negative):** Underperformance, missed some expectations, cautious or slightly negative guidance, identifiable risks or headwinds.
    - **0-29 (Strongly Negative):** Significant underperformance, highly pessimistic outlook, major operational issues, severe market challenges, high financial risk. Assign lower scores (0-10) for extremely dire situations.

    ### RULES:
    - Only use the content in the transcripts. No external assumptions or outside knowledge.
    - Be precise and concise in all sections.
    - Ensure your Sentiment Score directly reflects your Analysis and Sentiment Justification for that quarter.

    ### FINAL ANALYSIS SUMMARY:
    **[IMPORTANT: This section MUST be the absolute final output of your response. Do NOT include any other text after this block.]**
    Q4_Year1_Sentiment_Score: [Q4 of Year1 Sentiment Score numerical value]
    Q1_Year2_Sentiment_Score: [Q1 of Year2 Sentiment Score numerical value]
    Percentage_Change_Value: [e.g., +15.0% or -8.5% or N/A (Q4 of Year1 Sentiment Score was 0)]
    Final_Change_Summary: [A concise, one-sentence summary of the sentiment change, suitable for direct extraction.]
    """

    # --- Initialize results structures for PER SECTOR analysis ---
    # Will now store (percentage_change_value, market_cap) for each company
    sector_weighted_data = defaultdict(list) # {sector_name: [(percentage, market_cap), ...], ...}
    all_sector_analysis_results = {} # {sector_name: {ticker: {Q1_score, Q2_score, Percentage_Change}, ...}, ...}

    # --- Iterate through the DataFrame to get both ticker AND sector for each company ---
    # Now iterating through the pre-screened companies with Market_Cap
    for index, row in companies_df_for_analysis.iterrows():
        ticker = row['Symbol']
        sector = row['GICS_Sector'] 
        market_cap = row['MarketCap'] # Retrieve market cap

        print(f"\n======== Processing Company: {ticker} (Sector: {sector}, Market Cap: ${market_cap:,.0f}) ========\n")

        # Ensure the sector entry exists in the results dictionaries
        if sector not in all_sector_analysis_results:
            all_sector_analysis_results[sector] = {}

        # Vectorstore setup (remains largely the same)
        print("\nSetting up/Loading Chroma vector store...")
        vectorstore = Chroma(
            persist_directory=TRANSCRIPT_CHROMA_PATH,
            embedding_function=embeddings_model,
            collection_name=COLLECTION_NAME
        )
        initial_db_count = vectorstore._collection.count()
        print(f"Initial document count in the vector store: {initial_db_count}")
        
        full_parsed_transcripts = {} # Reset for each company's transcripts 
        
        transcript_sources = [
            {"ticker": ticker, "quarter": first_transcript_quarter, "year": first_transcript_year, "source_id": f"{ticker}_Q{first_transcript_quarter}_{first_transcript_year}"},
            {"ticker": ticker, "quarter": second_transcript_quarter, "year": second_transcript_year, "source_id": f"{ticker}_Q{second_transcript_quarter}_{second_transcript_year}"},
        ]

        for source_info in transcript_sources:
            current_ticker = source_info["ticker"]
            quarter = source_info["quarter"]
            year = source_info["year"]
            source_id = source_info["source_id"]

            print(f"\n--- Processing {source_id} ---")
            
            existing_chunks_for_source = vectorstore.get(
                where={"source": source_id},
                include=['documents', 'metadatas']
            )
            
            if existing_chunks_for_source and existing_chunks_for_source['documents']:
                print(f"✅ Transcript for {source_id} already in database. Reconstructing full content from {len(existing_chunks_for_source['documents'])} chunks.")
                
                reconstructed_full_content = ""
                sorted_retrieved_items = sorted(
                    zip(existing_chunks_for_source['documents'], existing_chunks_for_source['metadatas']),
                    key=lambda x: int(x[1]['id'].split('_')[3]) if 'id' in x[1] and len(x[1]['id'].split('_')) > 3 else 0
                )

                for doc_content, doc_metadata in sorted_retrieved_items:
                    reconstructed_full_content += doc_content + " " 

                full_parsed_transcripts[source_id] = reconstructed_full_content.strip()
                print("Preview of reconstructed content (first 500 chars):")
                print(full_parsed_transcripts[source_id][:500])

            else:
                print(f"Transcript for {source_id} NOT found in database. Pulling from API sources...")
                parsed_transcript_content = fetch_transcript_from_api(current_ticker, year, quarter, API_key) 

                if parsed_transcript_content: # This check is reliable now
                    full_parsed_transcripts[source_id] = parsed_transcript_content
                    print(f"Successfully retrieved new content for {source_id}. Preview (first 500 chars):")
                    print(full_parsed_transcripts[source_id][:500])

                    docs = text_splitter.create_documents([parsed_transcript_content], metadatas=[{"source": source_id}])
                    print(f"Transcript {source_id} split into {len(docs)} chunks for database storage.")

                    processed_chunks_with_ids = []
                    for j, chunk in enumerate(docs):
                        chunk_with_id = generate_document_id(chunk, j)
                        processed_chunks_with_ids.append(chunk_with_id)

                    all_existing_ids_in_db = set(vectorstore.get(include=[])['ids'])

                    new_chunks_to_add_to_db = []
                    new_chunk_ids_to_add_to_db = []
                    for chunk in processed_chunks_with_ids:
                        if chunk.metadata["id"] not in all_existing_ids_in_db:
                            new_chunks_to_add_to_db.append(chunk)
                            new_chunk_ids_to_add_to_db.append(chunk.metadata["id"])
                    
                    if new_chunks_to_add_to_db:
                        print(f"👉 Adding {len(new_chunks_to_add_to_db)} new chunks for {source_id} to the vector store in batches.")
                        add_documents_in_batches(vectorstore, new_chunks_to_add_to_db, new_chunk_ids_to_add_to_db)
                    else:
                        print(f"No new chunks to add for {source_id}. All relevant chunks likely already existed.")
                    
                else:
                    print(f"No valid parsed transcript for {source_id}. Skipping database ingestion and LLM processing for this quarter.")

        print(f"\nVector database operations complete for collection: 'transcripts'")
        print(f"Vector database path: {TRANSCRIPT_CHROMA_PATH}")
        
        final_db_count = vectorstore._collection.count()
        print(f"Final document count in the vector store: {final_db_count}")
        
        # --- LLM Processing (Sentiment Analysis) ---
        print(f"\n--- Performing LLM Sentiment Analysis for {ticker} (Sector: {sector}) ---")

        transcript_first_quarter_content = full_parsed_transcripts.get(f"{ticker}_Q{first_transcript_quarter}_{first_transcript_year}")
        transcript_second_quarter_content = full_parsed_transcripts.get(f"{ticker}_Q{second_transcript_quarter}_{second_transcript_year}")

        if not transcript_first_quarter_content or not transcript_second_quarter_content:
            print(f"\nCRITICAL: One or both full transcripts for {ticker} were not available despite pre-screening. Skipping this company for analysis.")
            continue 

        print(f"\n--- DEBUG: Transcript Content Preview for {ticker} ---")
        print(f"Content for {ticker} Q{first_transcript_quarter} {first_transcript_year}: {transcript_first_quarter_content[:500]}...\n") 
        print(f"Content for {ticker} Q{second_transcript_quarter} {second_transcript_year}: {transcript_second_quarter_content[:500]}...\n") 
        print("--- END DEBUG ---\n")

        full_human_message_content = f"""
        Here are the earnings call transcripts for (ticker: {ticker}) for Q{first_transcript_quarter} of {first_transcript_year} and Q{second_transcript_quarter} of {second_transcript_year} respectively.

        --- Q{first_transcript_quarter} of {first_transcript_year} Transcript START ---
        {transcript_first_quarter_content}
        --- Q{first_transcript_quarter} of {first_transcript_year} Transcript END ---

        --- Q{second_transcript_quarter} of {second_transcript_year} Transcript START ---
        {transcript_second_quarter_content}
        --- Q{second_transcript_quarter} of {second_transcript_year} Transcript END ---

        Please:
        1. Analyze Q{first_transcript_quarter} of {first_transcript_year} first using the specified format.
        2. Analyze Q{second_transcript_quarter} of {second_transcript_year} next using the same format.
        3. Then compare them and compute the **percentage change in sentiment score**.

        Remember: Follow the exact structure and formatting. Do not mix both quarters in one analysis.
        """

        messages = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt_content),
                HumanMessage(content=full_human_message_content),
            ]
        )

        llm_response = llm.invoke(messages.format_messages())

        print(f"\n\n--------------------- LLM Response for {ticker} (Sector: {sector}) ---------------------\n\n")
        print(llm_response.content)

        # --- Post-processing to extract scores and percentage change ---
        # Initialize scores
        first_quarter_score = None
        second_quarter_score = None
        percentage_change_str = None


        # STRATEGY 1: Extract sentiment scores from '**Qx of YYYY Sentiment Score:** NN'
        score_matches = re.findall(r"\*\*([\w_]+):\*\*\s*(\d+)", llm_response.content, re.DOTALL)


        print(f"DEBUG: Found score_matches: {score_matches}")


        # Now, process the score_matches to assign to first_quarter_score and second_quarter_score
        for key, score_str in score_matches:
            if key == 'Q4_Year1_Sentiment_Score':
                first_quarter_score = float(score_str)
            elif key == 'Q1_Year2_Sentiment_Score':
                second_quarter_score = float(score_str)


        print(f"DEBUG: first_quarter_score (Q4_Year1): {first_quarter_score}")
        print(f"DEBUG: second_quarter_score (Q1_Year2): {second_quarter_score}")


        # STRATEGY 2: Extract from LaTeX-style formula
        if first_quarter_score is None or second_quarter_score is None:
            calculation_match_latex = re.search(r'\\frac{([+\-]?\d+\.?\d*)\s*-\s*([+\-]?\d+\.?\d*)}', llm_response.content)
            if calculation_match_latex:
                try:
                    second_quarter_score = float(calculation_match_latex.group(1))
                    first_quarter_score = float(calculation_match_latex.group(2))
                    print(f"DEBUG: Extracted from LaTeX: First Q = {first_quarter_score}, Second Q = {second_quarter_score}")
                except ValueError:
                    print("Warning: LaTeX format found but score parsing failed.")


        # STRATEGY 3: Try to extract explicit percentage change
        pct_match = re.search(r"Percentage Change in Sentiment Score:\s*([^\n]*)", llm_response.content)
        if pct_match:
            captured_value = pct_match.group(1).strip()
            match_number = re.search(r"([\+\-]?\d+\.?\d+%)", captured_value)
            if match_number:
                percentage_change_str = match_number.group(1)
                print(f"DEBUG: Found explicit percentage change: {percentage_change_str}")
            elif "N/A" in captured_value:
                percentage_change_str = captured_value
                print(f"DEBUG: Found N/A percentage change: {percentage_change_str}")


        # STRATEGY 4: Fallback calculation
        if percentage_change_str is None:
            if first_quarter_score is not None and second_quarter_score is not None:
                if first_quarter_score == 0:
                    percentage_change_str = "N/A (Score was 0)"
                    print("DEBUG: First quarter score was 0, cannot compute percentage change.")
                else:
                    try:
                        percentage_change_val = ((second_quarter_score - first_quarter_score) / first_quarter_score) * 100
                        percentage_change_str = f"{percentage_change_val:.2f}%"
                        print(f"DEBUG: Fallback calculated percentage change: {percentage_change_str}")
                    except Exception as e:
                        percentage_change_str = f"Error: {str(e)}"
                        print("DEBUG: Exception during fallback calculation.")
            else:
                percentage_change_str = "Not available (scores missing)"
                print("DEBUG: Fallback failed due to missing scores.")


        # --- Store results for the current company within its sector ---
        all_sector_analysis_results[sector][ticker] = {
            f"Q{first_transcript_quarter}_{first_transcript_year}_Sentiment_Score": first_quarter_score,
            f"Q{second_transcript_quarter}_{second_transcript_year}_Sentiment_Score": second_quarter_score,
            "Percentage_Change": percentage_change_str
        }

        # --- Add valid percentage changes AND market cap to the list for its specific sector for weighted averaging ---
        if percentage_change_str and \
        percentage_change_str not in [f"N/A (Q{first_transcript_quarter} Sentiment Score was 0)", "Not available (scores missing for calculation)", "Error calculating"]:
            try:
                # Remove '%' and convert to float for calculation
                percentage_value = float(percentage_change_str.replace('%', ''))
                # Store tuple of (percentage_value, market_cap)
                sector_weighted_data[sector].append((percentage_value, market_cap)) 
            except ValueError:
                print(f"Warning: Could not parse percentage change '{percentage_change_str}' for {ticker} for averaging.")
        
    # --- START OF NEW LOGIC FOR APPENDING COLUMNS ---
    csv_records_for_output = []
    
    # Construct the dynamic column header based on user input for the desired format
    dynamic_column_header = f"Q{first_transcript_quarter}_{first_transcript_year}-Q{second_transcript_quarter}_{second_transcript_year}"

    # Get a list of all unique sectors from the initial DataFrame, to ensure all sectors are covered
    all_gics_sectors = companies_df_for_analysis['GICS_Sector'].unique()

    for sector_name in sorted(all_gics_sectors): # Iterate through all possible sectors
        companies_data = sector_weighted_data.get(sector_name, []) # Get data for this sector, default to empty list if none
            
        weighted_avg_sentiment_change = None
        num_tickers_included = len(companies_data) # Count of companies included with valid data (that had valid % change)

        if companies_data: # Only proceed if there's actual data to weight
            total_market_cap_in_sector = sum(item[1] for item in companies_data) # sum of market_caps for companies *with valid sentiment change*

            if total_market_cap_in_sector > 0:
                weighted_sum_of_changes = sum(item[0] * (item[1] / total_market_cap_in_sector) for item in companies_data)
                weighted_avg_sentiment_change = round(weighted_sum_of_changes, 1) # Round to 1 decimal place

        # Format the combined value string exactly as requested: "(value, count)"
        if weighted_avg_sentiment_change is not None:
            combined_value_string = f"({weighted_avg_sentiment_change}, {num_tickers_included})"
        else:
            combined_value_string = f"(N/A, {num_tickers_included})" # Use N/A if sentiment change couldn't be calculated

        # Add to the list with ONLY the two desired columns for the current quarter's data
        csv_records_for_output.append({
            'Sector': sector_name,
            dynamic_column_header: combined_value_string
        })
    
    csv_file_name = 'data/sector_sentiment_analysis_summary.csv'
    new_quarter_df = pd.DataFrame(csv_records_for_output)

    if os.path.exists(csv_file_name):
        print(f"\nExisting CSV '{csv_file_name}' found. Merging new quarter data as a column.")
        existing_df = pd.read_csv(csv_file_name)
        
        # Merge the existing DataFrame with the new quarter's data
        # 'outer' merge ensures all sectors from both are kept.
        # This will add the new column or update if it already exists (if dynamic_column_header matches an existing one)
        combined_df = pd.merge(existing_df, new_quarter_df, on='Sector', how='outer')

        # Fill any NaNs that might have appeared for existing columns (for new sectors) with "(N/A, 0)"
        for col in combined_df.columns:
            if col != 'Sector' and combined_df[col].isnull().any():
                combined_df[col].fillna("(N/A, 0)", inplace=True) 

        # Reorder columns: 'Sector' first, then existing columns, then the new dynamic column.
        ordered_columns = ['Sector']
        # Add existing columns from previous file, excluding the current dynamic column
        for col in existing_df.columns:
            if col != 'Sector' and col != dynamic_column_header:
                ordered_columns.append(col)
        # Add the current dynamic column (it will be added only once if it exists or is new)
        if dynamic_column_header in combined_df.columns and dynamic_column_header not in ordered_columns:
            ordered_columns.append(dynamic_column_header)

        # Handle any other new columns that might have been introduced during merge but not explicitly ordered (unlikely but safe)
        # This loop ensures that any columns added by 'outer' merge not in ordered_columns (very rare for this use case) are included.
        for col in combined_df.columns:
            if col not in ordered_columns:
                ordered_columns.append(col)

        # Ensure all columns in ordered_columns exist in combined_df before reindexing
        ordered_columns = [col for col in ordered_columns if col in combined_df.columns]
        combined_df = combined_df[ordered_columns]
        
        print(f"✅ Appended new column '{dynamic_column_header}' to '{csv_file_name}'.")

    else:
        print(f"\nNo existing CSV '{csv_file_name}' found. Creating new file with data for '{dynamic_column_header}'.")
        combined_df = new_quarter_df
    
    # Write the combined DataFrame back to the CSV (mode='w' to overwrite the file with the new structure)
    if not combined_df.empty:
        combined_df.to_csv(csv_file_name, mode='w', index=False, header=True)
        print(f"✅ CSV '{csv_file_name}' updated successfully.")
    else:
        print("\nNo data to write to CSV after combining.")

    # --- END OF NEW LOGIC FOR APPENDING COLUMNS ---

    # --- Final: Calculate and Print Weighted Average Percentage Change for EACH Sector (console output, for debugging/visibility) ---
    # This section remains for console readability, as it was not part of the specific CSV format request
    print("\n\n======== Sector-Wide Analysis Summary (Console) ========\n")

    if sector_weighted_data:
        for sector_name in sorted(sector_weighted_data.keys()): 
            companies_data = sector_weighted_data[sector_name]
            
            if companies_data: 
                # Calculate total market cap for *these specific companies* for weighting
                total_market_cap_in_sector = sum(item[1] for item in companies_data) # sum of market_caps

                if total_market_cap_in_sector > 0:
                    weighted_sum_of_changes = sum(item[0] * (item[1] / total_market_cap_in_sector) for item in companies_data)
                    print(f"Weighted Average Percentage Change in Sentiment Score for {sector_name}: {weighted_sum_of_changes:.1f}% (based on {len(companies_data)} companies)")
                else:
                    print(f"No valid market cap data for {sector_name} to calculate a weighted average.")
            else:
                print(f"No valid percentage changes for {sector_name} to calculate a weighted average.")
    else:
        print("No valid percentage changes were extracted for any sector to calculate a weighted average.")

    print("\n--- Detailed Company Results Per Sector (Console) ---")
    for sector_name in sorted(all_sector_analysis_results.keys()): 
        print(f"\nSector: {sector_name}")
        companies_in_sector = all_sector_analysis_results[sector_name]
        if not companies_in_sector:
            print(f"  No company results for {sector_name}.")
            continue
        for ticker_key in sorted(companies_in_sector.keys()): 
            results = companies_in_sector[ticker_key]
            print(f"  Company: {ticker_key}")
            print(f"    Q{first_transcript_quarter} {first_transcript_year} Sentiment Score: {results.get(f'Q{first_transcript_quarter}_{first_transcript_year}_Sentiment_Score', 'N/A')}")
            print(f"    Q{second_transcript_quarter} {second_transcript_year} Sentiment Score: {results.get(f'Q{second_transcript_quarter}_{second_transcript_year}_Sentiment_Score', 'N/A')}")
            print(f"    Percentage Change: {results.get('Percentage_Change', 'N/A')}")


if __name__ == "__main__":
    get_sentiment()