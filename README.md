# Policy RAG System

Optimized embedding system for insurance policy document processing and querying.

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv venv  # Use python3 on Mac/Linux
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Process all PDFs and create embeddings
python main.py process

# Search documents
python main.py search "knee surgery coverage"

# Process insurance query with decision logic
python main.py query "46M knee surgery Pune 3-month policy"

# View system statistics
python main.py stats
```

## Commands Explained

### `python main.py process`
- **Purpose**: Process all PDF files and create embeddings
- **What it does**: Extracts text from PDFs, splits into chunks, generates vectors, stores in database
- **When to use**: First time setup or when adding new documents
- **Output**: Number of files processed and chunks created

### `python main.py search "query"`
- **Purpose**: Find documents similar to your search query
- **What it does**: Converts query to vector, finds similar document sections
- **When to use**: When you want to find specific policy information
- **Output**: Ranked list of relevant document sections

### `python main.py query "insurance query"`
- **Purpose**: Process insurance claims with decision logic
- **What it does**: Searches documents + analyzes coverage/exclusions to make decisions
- **When to use**: For insurance claim processing and coverage decisions
- **Output**: Decision (covered/excluded), confidence score, justification

### `python main.py stats`
- **Purpose**: View system statistics
- **What it does**: Shows database stats, document counts, section distribution
- **When to use**: To check system health and data overview
- **Output**: Vector counts, files processed, section breakdown

## Features

- **Smart Document Processing**: Semantic chunking by coverage sections
- **Free Vector Storage**: Chroma + Sentence-BERT embeddings
- **Batch Processing**: Optimized for multiple documents
- **Metadata Tracking**: SQLite for structured data
- **Insurance-Specific Logic**: Coverage/exclusion classification
- **CLI Interface**: Easy testing and operations

## Architecture

```
├── src/
│   ├── document_processor.py   # PDF processing & chunking
│   ├── embedding_engine.py     # Main engine & query processing
│   └── vector_store.py         # Chroma + SQLite integration
├── data/
│   ├── processed/              # Processed documents
│   └── embeddings/             # Vector database
├── assets/                     # Source PDF files
├── config.py                   # Configuration
└── main.py                     # CLI interface

