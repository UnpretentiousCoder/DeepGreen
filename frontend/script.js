// script.js
// ===== Configuration =====
const API_BASE_URL = ''; // Your backend FastAPI URL

// ===== UI Elements =====
const elements = {
    // Chat elements
    chatContainer: document.getElementById('chat-container'),
    messageInput: document.getElementById('message-input'),
    sendButton: document.getElementById('send-button'),

    // Upload elements
    fileInput: document.getElementById('fileInput'),
    uploadArea: document.querySelector('.upload-area'),
    fileStatus: document.querySelector('.file-status'),
    uploadButton: document.getElementById('upload-button'),
    ingestButton: document.getElementById('ingest-button'),
    ingestFileNameDisplay: document.getElementById('ingestFileNameDisplay'),

    // Navigation
    navItems: document.querySelectorAll('.nav-item'),
    pages: document.querySelectorAll('.page'),

    // Status
    statusMessagesContainer: document.getElementById('status-messages-container')
};

// dropdown for “Comparing”
const periodSelect = document.getElementById('period-select');
const PERIOD_OPTIONS = [
    "Q2_2025 vs Q1_2025",
    "Q1_2025 vs Q4_2024",
    "Q4_2024 vs Q3_2024", //Q1_2025-Q2_2025 NEED TO ADD NEW PERIODFS IF I REBUILD
    //"Q3_2024 vs Q2_2024",
    // add more historic pairs here
];

function populatePeriodDropdown(options) {
    periodSelect.innerHTML = '';
    options.forEach(optText => {
        const opt = document.createElement('option');
        opt.value = optText;
        opt.textContent = optText;
        periodSelect.appendChild(opt);
    });
}


// ===== State Management =====
let currentFileToUpload = null;

// ===== UI Helper Functions =====
function displayGlobalStatusMessage(message, type = 'info', showSpinner = false) {
    if (!elements.statusMessagesContainer) {
        console.warn("Status messages container not found!");
        return null;
    }

    const msgElement = document.createElement('div');
    msgElement.classList.add('status-message', type);

    if (showSpinner) {
        const spinnerDiv = document.createElement('div');
        spinnerDiv.classList.add('spinner', 'inline-spinner');
        msgElement.appendChild(spinnerDiv);

        const textSpan = document.createElement('span');
        textSpan.textContent = ` ${message}`;
        msgElement.appendChild(textSpan);
    } else {
        msgElement.textContent = message;
    }

    msgElement.style.animation = 'fadeInSlideDown 0.3s forwards';
    elements.statusMessagesContainer.appendChild(msgElement);

    if (!showSpinner) {
        setTimeout(() => {
            msgElement.style.animation = 'fadeOut 0.5s forwards';
            msgElement.addEventListener('animationend', () => {
                msgElement.remove();
            }, { once: true });
        }, 5000);
    }

    return msgElement;
}

function addMessage(sender, text) {
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');

    if (sender === 'user') {
        messageDiv.classList.add('user-message');
        messageDiv.textContent = text;
    } else if (sender === 'ai') {
        messageDiv.classList.add('ai-message');
        // Remove <think> tags if present
        let cleanText = text;
        const thinkTagStart = text.indexOf('<think>');
        const thinkTagEnd = text.indexOf('</think>');

        if (thinkTagStart !== -1 && thinkTagEnd !== -1) {
            cleanText = text.substring(0, thinkTagStart) + text.substring(thinkTagEnd + '</think>'.length);
        }

        messageDiv.innerHTML = marked.parse(cleanText);
    } else if (sender === 'loading') {
        messageDiv.classList.add('loading-message');
        messageDiv.innerHTML = `
            <div class="loading-dots">
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
            </div>
        `;
    }

    elements.chatContainer.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

function scrollToBottom() {
    if (elements.chatContainer) {
        elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    }
}

function enableInputs(enable) {
    elements.messageInput.disabled = !enable;
    elements.sendButton.disabled = !enable;

    if (elements.uploadArea) {
        elements.uploadArea.style.pointerEvents = enable ? 'auto' : 'none';
        elements.uploadArea.style.opacity = enable ? '1' : '0.7';
    }

    if (elements.fileInput) {
        elements.fileInput.disabled = !enable;
    }
}

// ===== Navigation =====
elements.navItems.forEach(item => {
    item.addEventListener('click', () => {
        const targetPageId = item.dataset.page;

        // Update nav items
        elements.navItems.forEach(nav => nav.classList.remove('active'));
        item.classList.add('active');

        // Update pages
        elements.pages.forEach(page => page.classList.remove('active'));
        const targetPage = document.getElementById(targetPageId);
        if (targetPage) {
            targetPage.classList.add('active');
        }

        // Scroll chat to bottom if navigating to chat
        if (targetPageId === 'chat-page') {
            scrollToBottom();
        }

        // Load sentiment data when navigating to sentiment page
        if (targetPageId === 'sentiment-page') {
            displayExistingSentimentData();
        }
    });
});

// ===== State Management =====
let currentFilesToUpload = []; // Changed from single file to array

// ===== File Handling =====
function handleFiles(files) {
    if (files.length === 0) {
        currentFilesToUpload = [];
        elements.fileStatus.textContent = 'No files chosen';
        elements.uploadButton.disabled = true;
        elements.ingestFileNameDisplay.textContent = 'all uploaded PDFs';
        return;
    }

    // Filter for PDF files only
    const pdfFiles = Array.from(files).filter(file => file.type === 'application/pdf');
    const invalidFiles = files.length - pdfFiles.length;

    if (pdfFiles.length === 0) {
        currentFilesToUpload = [];
        elements.fileStatus.textContent = 'Please select PDF files only.';
        elements.uploadButton.disabled = true;
        displayGlobalStatusMessage('Error: Only PDF files are allowed.', 'error');
        return;
    }

    currentFilesToUpload = pdfFiles;
    
    // Update status text for multiple files
    if (pdfFiles.length === 1) {
        elements.fileStatus.textContent = `File selected: ${pdfFiles[0].name}`;
        elements.ingestFileNameDisplay.textContent = pdfFiles[0].name;
    } else {
        elements.fileStatus.textContent = `${pdfFiles.length} files selected`;
        const fileNames = pdfFiles.map(f => f.name).join(', ');
        elements.ingestFileNameDisplay.textContent = `${pdfFiles.length} files: ${fileNames.length > 100 ? fileNames.substring(0, 100) + '...' : fileNames}`;
    }
    
    elements.uploadButton.disabled = false;
    
    // Show appropriate message
    if (invalidFiles > 0) {
        displayGlobalStatusMessage(`${pdfFiles.length} PDF files ready to upload. ${invalidFiles} non-PDF files ignored.`, 'warning');
    } else {
        displayGlobalStatusMessage(`Ready to upload ${pdfFiles.length} file(s)`, 'info');
    }
}

async function uploadFile() {
    if (!currentFilesToUpload || currentFilesToUpload.length === 0) {
        displayGlobalStatusMessage('No files selected for upload.', 'error');
        return;
    }

    const fileCount = currentFilesToUpload.length;
    const uploadMsg = displayGlobalStatusMessage(
        `Uploading ${fileCount} file(s)...`, 
        'info', 
        true
    );

    elements.uploadButton.disabled = true;
    elements.ingestButton.disabled = true;
    elements.messageInput.disabled = false;
    elements.sendButton.disabled = false;
    elements.fileInput.disabled = false;

    try {
        // Option 1: Upload all files in a single request
        const formData = new FormData();
        currentFilesToUpload.forEach((file, index) => {
            formData.append(`files`, file); // Use 'files' as the field name
        });

        const response = await fetch(`${API_BASE_URL}/api/uploadfiles/`, { // Note: changed endpoint to plural
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (response.ok) {
            displayGlobalStatusMessage(`${fileCount} file(s) uploaded successfully!`, 'success');
        } else {
            const errorDetail = data.detail || JSON.stringify(data);
            console.warn(`Bulk upload failed (HTTP ${response.status}), falling back to individual:`, errorDetail);
            throw new Error(`Bulk upload failed (HTTP ${response.status}), falling back to individual: ${errorDetail}`);
        }
    } catch (error) {
        //displayGlobalStatusMessage(`Error: ${error.message}`, 'error');
        console.warn("Caught an issue during bulk upload attempt (proceeding to individual uploads):", error.message);
        // Option 2: Fallback to individual uploads if bulk upload fails
        await uploadFilesIndividually();
    } finally {
        if (uploadMsg) uploadMsg.remove();
        elements.messageInput.disabled = false;
        elements.sendButton.disabled = false;
        elements.fileInput.disabled = false;
        if (currentFilesToUpload.length > 0) {
            elements.uploadButton.disabled = false;
        } else {
            elements.uploadButton.disabled = true;
        }
        elements.ingestButton.disabled = false;
    }
}
async function uploadFilesIndividually() {
    const uploadMsg = displayGlobalStatusMessage(
        `Uploading files individually...`, 
        'info', 
        true
    );

    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < currentFilesToUpload.length; i++) {
        const file = currentFilesToUpload[i];
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(`${API_BASE_URL}/api/uploadfile/`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (response.ok) {
                successCount++;
                console.log(`Successfully uploaded: ${file.name}`);
            } else {
                errorCount++;
                console.error(`Failed to upload ${file.name}:`, data.detail || data);
            }
        } catch (error) {
            errorCount++;
            console.error(`Error uploading ${file.name}:`, error.message);
        }
    }

    if (uploadMsg) uploadMsg.remove();

    if (successCount > 0 && errorCount === 0) {
        displayGlobalStatusMessage(`All ${successCount} files uploaded successfully!`, 'success');
    } else if (successCount > 0 && errorCount > 0) {
        displayGlobalStatusMessage(`${successCount} files uploaded, ${errorCount} failed`, 'warning');
    } else {
        displayGlobalStatusMessage(`All uploads failed`, 'error');
    }
}

async function startIngestion() {
    const ingestMsg = displayGlobalStatusMessage('Starting ingestion process...', 'info', true);

    elements.ingestButton.disabled = true;
    elements.uploadButton.disabled = true;
    elements.messageInput.disabled = false;
    elements.sendButton.disabled = false;
    elements.fileInput.disabled = false;

    try {
        const response = await fetch(`${API_BASE_URL}/api/ingest/`, {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });

        const data = await response.json();

        if (response.ok) {
            displayGlobalStatusMessage(`Ingestion complete! ${data.message || ''}`, 'success');
            elements.ingestFileNameDisplay.textContent = 'all uploaded PDFs';
        } else {
            const errorDetail = data.detail || JSON.stringify(data);
            displayGlobalStatusMessage(`Ingestion failed: ${errorDetail}`, 'error');
            throw new Error(`Ingestion failed: ${errorDetail}`);
        }
    } catch (error) {
        displayGlobalStatusMessage(`Error: ${error.message}`, 'error');
    } finally {
        if (ingestMsg) ingestMsg.remove();
        elements.messageInput.disabled = false;
        elements.sendButton.disabled = false;
        elements.fileInput.disabled = false;
        elements.ingestButton.disabled = false;
        elements.uploadButton.disabled = true;
    }
}

async function sendMessage() {
    const query = elements.messageInput.value.trim();

    if (!query) {
        displayGlobalStatusMessage('Please enter a question.', 'info');
        return;
    }

    addMessage('user', query);
    elements.messageInput.value = '';
    const loadingMsg = addMessage('loading', '');
    enableInputs(false);

    try {
        const response = await fetch(`${API_BASE_URL}/api/query/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();
        loadingMsg.remove();
        console.log(typeof data.response);
        console.log(typeof data.sources);
        if (response.ok) {
            addMessage('ai', data.response);
            addMessage('ai', "From: " + data.sources)
        } else {
            displayGlobalStatusMessage(`Query failed: ${data.detail || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        loadingMsg.remove();
        displayGlobalStatusMessage(`Error: ${error.message}`, 'error');
    } finally {
        enableInputs(true);
    }
}

// Updated frontend function to accept period parameter
async function displayExistingSentimentData(selectedPeriod = null) {
    const sentimentMsg = displayGlobalStatusMessage('Loading existing sentiment data...', 'info', true);

    try {
        // Build URL with period parameter if provided
        let url = `${API_BASE_URL}/api/sentiment/display/`;
        if (selectedPeriod) {
            url += `?period=${encodeURIComponent(selectedPeriod)}`;
        }

        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        });     

        const data = await response.json();
        console.log('Full API response:', data);
        console.log('Sectors data:', data.data?.sectors);

        if (data.status === 'success') {
            // Display the data in your UI
            updateSentimentDisplay(data.data);
            displayGlobalStatusMessage(`Sentiment analysis complete!`, 'success');
        } else {
            throw new Error(data.message || 'Failed to load sentiment data');
        }
    } catch (error) {
        console.error('Error loading sentiment data:', error);
        displayGlobalStatusMessage(`Error loading sentiment data: ${error.message}`, 'error');
        showMockSentimentData();
    }
    finally {
        // clear the loading message
        if (sentimentMsg) sentimentMsg.remove();
    }
}

// ===== Sentiment Analysis Functions =====
async function fetchNewSentimentData() { //for new sentiment data
    // show loading message
    const sentimentMsg = displayGlobalStatusMessage('Fetching new sentiment data...', 'info', true);

    try {
        // determine URL based on whether we're creating a new comparison or refreshing an existing one
        const response = await fetch(`${API_BASE_URL}/api/sentiment/get_newest/`, { //NOTE THE REFRESH ISNT DONE YET, this is also WRONG
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            signal: AbortSignal.timeout(21600000)
        });
        const data = await response.json();
        console.log('Full API response:', data);
        console.log('Sectors data:', data.data?.sectors);

        if (response.ok && data.status === 'success') {
            // if backend returned a new list of periods, update the dropdown
            if (data.data.availablePeriods) {
                populatePeriodDropdown(data.data.availablePeriods); 
                periodSelect.value = data.data.period;
            }
            displayGlobalStatusMessage('Sentiment data loaded!', 'success');
            updateSentimentDisplay(data.data);
        } 
        else if (data.status === 'error') {
        // Handle error status (e.g., date restrictions, other errors)
        const errorMessage = data.message || 'Failed to fetch sentiment data';
        displayGlobalStatusMessage(errorMessage, 'error');
        await displayExistingSentimentData();
        }
        else {
            const errorDetail = data.detail || data.message || JSON.stringify(data);
            displayGlobalStatusMessage(`Failed to fetch sentiment data: ${errorDetail}`, 'error');
            await displayExistingSentimentData();
        }
    } catch (error) {
        displayGlobalStatusMessage(`Error: ${error.message}`, 'error');

    } finally {
        // clear the loading message
        if (sentimentMsg) sentimentMsg.remove();
    }
}

async function refreshSentimentData() { //for refreshing existing sentiment data
    const sentimentMsg = displayGlobalStatusMessage('Refreshing current sentiment data...', 'info', true);
        try {
        const response = await fetch(`${API_BASE_URL}/api/sentiment/refresh/`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            signal: AbortSignal.timeout(21600000)
        });     

        const data = await response.json();
        console.log('Full API response:', data);
        console.log('Sectors data:', data.data?.sectors);

        if (response.ok && data.status === 'success') {
            // Display the data in your UI
            updateSentimentDisplay(data.data);
            displayGlobalStatusMessage(`Sentiment analysis complete!`, 'success');
        } else {
            throw new Error(data.message || 'Failed to load sentiment data');
        }
    } catch (error) {
        console.error('Error loading sentiment data:', error);
        displayGlobalStatusMessage(`Error loading sentiment data: ${error.message}`, 'error');
        showMockSentimentData();
    }
    finally {
        // clear the loading message
        if (sentimentMsg) sentimentMsg.remove();
    }
}

function updateSentimentDisplay(data) {
    // Update period info
    if (data.period) {
        periodSelect.value = data.period;
    }

    // Helper: parse a "(change, count)" tuple
    function parseSentimentValue(valueStr) {
        try {
            const cleanStr = valueStr.replace(/[()]/g, '');
            const parts = cleanStr.split(',').map(part => part.trim());
            const changeStr = parts[0];
            const transcriptCount = parseInt(parts[1], 10) || 0;

            if (changeStr === 'N/A' || changeStr === 'null') {
                return { score: 0, transcriptsAnalyzed: transcriptCount, change: 0, isNA: true };
            }

            const changePercent = parseFloat(changeStr) || 0;
            return {
                score: Math.abs(changePercent),
                transcriptsAnalyzed: transcriptCount,
                change: changePercent,
                isNA: false
            };
        } catch (error) {
            console.error('Error parsing sentiment value:', valueStr, error);
            return { score: 0, transcriptsAnalyzed: 0, change: 0, isNA: true };
        }
    }

    // Update each sector card
    Object.entries(data.sectors).forEach(([sectorKey, sectorValueStr]) => {
        const card = document.querySelector(`[data-sector="${sectorKey}"]`);
        if (!card) return;

        const sectorData = parseSentimentValue(sectorValueStr);

        const scoreEl = card.querySelector('.score-value');
        const arrowEl = card.querySelector('.change-arrow');
        const changeValEl = card.querySelector('.change-value');
        const scoreCont = card.querySelector('.sentiment-score');
        const changeCont = card.querySelector('.score-change');

        scoreCont.classList.remove('loading');

        if (sectorData.isNA) {
            scoreEl.textContent = 'N/A';
            arrowEl.textContent = '→';
            arrowEl.className = 'change-arrow neutral';
            changeValEl.textContent = 'N/A';
            changeCont.className = 'score-change neutral';
        } else {
            // Show change percentage as the main score
            scoreEl.textContent = sectorData.change.toFixed(1) + '%';

            const change = sectorData.change;
            const pct = Math.abs(change).toFixed(1);

            if (change > 0) {
                arrowEl.textContent = '↑';
                arrowEl.className = 'change-arrow up';
                changeValEl.textContent = `+${pct}%`;
                changeCont.className = 'score-change positive';
            } else if (change < 0) {
                arrowEl.textContent = '↓';
                arrowEl.className = 'change-arrow down';
                changeValEl.textContent = `-${pct}%`;
                changeCont.className = 'score-change negative';
            } else {
                arrowEl.textContent = '→';
                arrowEl.className = 'change-arrow neutral';
                changeValEl.textContent = `${pct}%`;
                changeCont.className = 'score-change neutral';
            }
        }

        // Transcripts-count badge
        let counter = card.querySelector('.transcript-count');
        if (!counter) {
            counter = document.createElement('div');
            counter.className = 'transcript-count';
            counter.innerHTML = '<span class="count-value"></span> transcripts analyzed';
            card.appendChild(counter);
        }
        counter.querySelector('.count-value').textContent = sectorData.transcriptsAnalyzed;
    });
}


function showMockSentimentData() {
    // base sentiment scores + changes
    const baseSectors = {
        "energy": { score: 0.72, change: 5.2 },
        "materials": { score: 0.65, change: -2.1 },
        "industrials": { score: 0.68, change: 3.4 },
        "consumer-discretionary": { score: 0.61, change: -1.8 },
        "consumer-staples": { score: 0.70, change: 0.0 },
        "healthcare": { score: 0.75, change: 4.7 },
        "financials": { score: 0.69, change: 2.3 },
        "it": { score: 0.78, change: 8.1 },
        "communication": { score: 0.64, change: -3.5 },
        "utilities": { score: 0.66, change: 1.2 },
        "real-estate": { score: 0.62, change: -4.3 }
    };

    // assign each sector a random transcriptsAnalyzed between 50 and 200
    const sectorsWithCounts = {};
    Object.entries(baseSectors).forEach(([key, val]) => {
        sectorsWithCounts[key] = {
            ...val,
            transcriptsAnalyzed: Math.floor(Math.random() * 151) + 50
        };
    });

    const mockData = {
        period: "Q1 2025 vs Q4 2024",
        sectors: sectorsWithCounts
    };

    updateSentimentDisplay(mockData);
}

// ===== Event Listeners =====
// Chat
elements.sendButton.addEventListener('click', sendMessage);
elements.messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Upload
elements.uploadButton.addEventListener('click', uploadFile);
elements.ingestButton.addEventListener('click', startIngestion);

// File input
elements.fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

// Drag and drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    elements.uploadArea.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
    });
});

['dragenter', 'dragover'].forEach(eventName => {
    elements.uploadArea.addEventListener(eventName, () => {
        elements.uploadArea.classList.add('dragover');
    });
});

['dragleave', 'drop'].forEach(eventName => {
    elements.uploadArea.addEventListener(eventName, () => {
        elements.uploadArea.classList.remove('dragover');
    });
});

elements.uploadArea.addEventListener('drop', (e) => {
    handleFiles(e.dataTransfer.files);
});

elements.uploadArea.addEventListener('click', () => {
    elements.fileInput.click();
});

// ===== Initialize =====
document.addEventListener('DOMContentLoaded', () => {
    // 1) Populate the “Comparing” dropdown and set default
    populatePeriodDropdown(PERIOD_OPTIONS);
    periodSelect.value = PERIOD_OPTIONS[0];

    // 2) Add event listener for period dropdown changes
    periodSelect.addEventListener('change', function() {
        const selectedPeriod = this.value;
        console.log('Period changed to:', selectedPeriod);
        
        // Load sentiment data for the selected period
        if (selectedPeriod) {
            displayExistingSentimentData(selectedPeriod);
        }
    });

    // 2) Set default page
    const defaultNavItem = document.querySelector('.nav-item[data-page="chat-page"]');
    if (defaultNavItem) {
        defaultNavItem.click();
    }

    // 3) Initialize Upload & Ingest UI state
    elements.uploadButton.disabled = true;
    elements.fileStatus.textContent = 'No file chosen';
    elements.ingestFileNameDisplay.textContent = 'all uploaded PDFs';

    // 4) Hook up the refresh button to show our custom modal
    const refreshButton = document.getElementById('refresh-sentiment');
    const refreshModal = document.getElementById('refresh-modal');
    const refreshCurrentBtn = document.getElementById('refresh-current');
    const startLatestBtn = document.getElementById('start-latest');
    const cancelRefreshBtn = document.getElementById('cancel-refresh');

    // helper to set all sentiment cards into loading state
    function setLoadingSentimentCards() {
        document.querySelectorAll('.sentiment-score').forEach(score => {
            score.classList.add('loading');
            score.querySelector('.score-value').textContent = '--';
            score.querySelector('.change-arrow').textContent = '-';
            score.querySelector('.change-value').textContent = '--%';
        });
    }

    refreshButton.addEventListener('click', () => {
        refreshModal.classList.add('show');
    });

    refreshCurrentBtn.addEventListener('click', () => {
        refreshModal.classList.remove('show');
        setLoadingSentimentCards();
        refreshSentimentData();
    });

    startLatestBtn.addEventListener('click', () => {
        refreshModal.classList.remove('show');
        setLoadingSentimentCards();
        fetchNewSentimentData();
    });

    cancelRefreshBtn.addEventListener('click', () => {
        refreshModal.classList.remove('show');
    });

    // 5) Click handlers for sector cards
    const sectorCards = document.querySelectorAll('.sector-card');
    sectorCards.forEach(card => {
        card.addEventListener('click', () => {
            const sectorName = card.querySelector('.sector-name').textContent;
            displayGlobalStatusMessage(`Selected: ${sectorName} sector`, 'info');
        });
    });

    // 6) Scroll chat to bottom (if chat page is active)
    scrollToBottom();
});
