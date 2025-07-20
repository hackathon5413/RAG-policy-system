# Insurance Policy RAG System

Clean, optimized RAG system for insurance policy documents with LangChain and Ollama integration.

## Setup

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama
# For macOS
brew install ollama
# OR download from https://ollama.ai/download

# For Linux
# curl -fsSL https://ollama.ai/install.sh | sh

ollama serve

ollama pull llama3.2:3b

# OR for more powerful hardware
# ollama pull llama3.1:8b

ollama serve
```

## Usage

```bash
# Process all PDFs in assets directory
python main.py ingest-dir ./assets

# Process single PDF
python main.py ingest policy.pdf

# Ask insurance questions
python main.py query "Is maternity covered?"

# Search documents
python main.py search "exclusions"

# Check statistics
python main.py stats
```

## Features

- **Smart Chunking**: LangChain with 600 char chunks, 100 char overlap
- **Section Classification**: Auto-detects coverage, exclusions, claims
- **Vector Search**: ChromaDB with sentence transformers
- **LLM Integration**: Ollama for natural language answers
- **Clean Architecture**: Single file, minimal dependencies


## Tech Stack

- **LangChain**: Document processing and chunking
- **ChromaDB**: Vector storage and similarity search  
- **Sentence Transformers**: Local embeddings (all-MiniLM-L6-v2)
- **Ollama**: Local LLM inference (Llama 3.2 3B)
- **Optimized**: Production-ready, minimal complexity
