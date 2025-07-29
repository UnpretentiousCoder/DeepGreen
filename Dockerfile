# Use a Python-specific slim image for Apple Silicon.
FROM python:3.10-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# --- 1. Install System Dependencies ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    curl \
    git \
    nginx \
    supervisor \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libmagic-dev \
    build-essential \
    python3-dev \
    gnupg \
    wget \
    xz-utils \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/log/nginx /var/log/supervisor

RUN addgroup --system --gid 101 nginx && \
    adduser --system --ingroup nginx --no-create-home --uid 101 --shell /bin/false nginx

# --- 2. Install and Link Newer SQLite3 (Still needed for ChromaDB) ---
ENV SQLITE_VERSION=3420000
ENV SQLITE_URL="https://www.sqlite.org/2023/sqlite-autoconf-${SQLITE_VERSION}.tar.gz"
ENV SQLITE_INSTALL_PATH=/usr/local

RUN wget -q ${SQLITE_URL} -O /tmp/sqlite-autoconf.tar.gz && \
    tar -xzf /tmp/sqlite-autoconf.tar.gz -C /tmp && \
    cd /tmp/sqlite-autoconf-${SQLITE_VERSION} && \
    ./configure --prefix=${SQLITE_INSTALL_PATH} && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/sqlite-autoconf* && \
    echo "${SQLITE_INSTALL_PATH}/lib" > /etc/ld.so.conf.d/sqlite3.conf && \
    ldconfig

ENV LD_LIBRARY_PATH="${SQLITE_INSTALL_PATH}/lib:${LD_LIBRARY_PATH}"

# --- 3. Configure Nginx and Frontend ---
RUN rm -f /var/www/html/index.nginx-debian.html
COPY ./frontend/index.html /var/www/html/
COPY ./frontend/script.js /var/www/html/
COPY ./frontend/style.css /var/www/html/
COPY ./frontend/nginx.conf /etc/nginx/nginx.conf
COPY ./frontend/default.conf /etc/nginx/conf.d/default.conf
RUN rm -f /etc/nginx/sites-enabled/default

# --- 4. Setup RAG App Environment ---
WORKDIR /app
COPY requirements.txt .
RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"
RUN /app/venv/bin/pip install --upgrade pip
RUN /app/venv/bin/pip install -r requirements.txt

# --- 5. Copy Application Files ---
COPY main.py .
COPY ingest_services.py .
COPY full_sentiment.py .
COPY refresh_sentiment.py .

# --- 6. Set Up Persistent Data Directories, prepares mount points ---
RUN mkdir -p /app/data \
             /app/chroma_db \
             /app/logs

# --- 7. Copy Supervisord Configuration ---
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# --- 8. Expose Ports ---
EXPOSE 80
EXPOSE 8000
# Port 11434 is no longer exposed by this container, as Ollama is on host

# --- 9. Define the Default Command ---
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]