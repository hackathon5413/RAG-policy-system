# Policy RAG System

Optimized embedding system for insurance policy document processing and querying.

## Quick Start

```bash
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
```

## Usage Examples

```python
from src.embedding_engine import PolicyQueryEngine

engine = PolicyQueryEngine()

# Process insurance query
result = engine.process_insurance_query("46M knee surgery Pune 3-month policy")

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']}")
print(f"Reasoning: {result['justification']['reasoning']}")
```

## System Requirements

- Python 3.8+
- 4GB RAM recommended
- 1GB disk space for embeddings
