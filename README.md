# tldr

## Setup

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd tldr

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

Ensure you have PostgreSQL with the pgvector extension available. Update your `.env` file with your database connection:

```env
APP_NAME="tldr"
DATABASE_URL="postgresql://username:password@host:port/database?sslmode=require"
MODEL_EMBEDDING="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
MODEL_LANGUAGE="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
```

### 3. Ollama Setup

Install Ollama and download the required models:

```bash
# Install Ollama (visit https://ollama.ai for installation instructions)

# Download required models
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

## Running the Application

### Standard Development Server

```bash
uvicorn src.main:app --reload
```

### With Custom Logging Configuration

```bash
uvicorn src.main:app --log-config log_conf.yaml --reload
```

The application will be available at `http://localhost:8000`