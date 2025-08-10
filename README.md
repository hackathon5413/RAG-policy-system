# LLM-Powered Intelligent Query-Retrieval System

**HackRX Challenge Solution**: FastAPI-based RAG system for processing documents and answering contextual queries in insurance, legal, HR, and compliance domains.

## 🚀 Setup Options

### � Option 1: Traditional Python Setup

Standard Python development approach:

```bash
# Clone the repository
git clone <repository>
cd policy-rag-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate     # On Linux/Mac
# OR
venv\Scripts\activate        # On Windows

# Install dependencies
pip install -r requirements.txt

# Set API key (use any of the 43 keys from docker-compose.yml)
export GEMINI_API_KEY_1="AIzaSyBuKHF-9oTwbCgWbY3B2-TmbJ6a1vd5iu4"

# Run the server
python server.py
```

### 🐳 Option 2: Docker (Containerized)

Super simple - one command setup:

```bash
# Clone and start
git clone <repository>
cd policy-rag-system

# One command - that's it!
docker-compose up --build
```

**Both options give you:**
- **API**: http://localhost:8080
- **Docs**: http://localhost:8080/docs
- **Health**: http://localhost:8080/health

## 📡 API Usage

### Test it with curl:
```bash
curl -X POST "http://localhost:8080/api/v1/hackrx/run" \
 -H "Content-Type: application/json" \
 -H "Accept: application/json" \
 -H "Authorization: Bearer 43e704a77310d35ab207cbb456481b2657cbf41a97bd1d2a3800e648acacb5c1" \
 -d '{
   "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
   "questions": [
       "What is the grace period for premium payment?",
       "What is the waiting period for pre-existing diseases?",
       "Does this policy cover maternity expenses?"
   ]
}'
```

### Response Format:
```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment...",
    "There is a waiting period of thirty-six (36) months...",
    "Yes, the policy covers maternity expenses..."
  ]
}
```

## 🛠️ Docker Commands

```bash
docker-compose up --build    # Start
docker-compose down          # Stop
docker-compose logs -f       # View logs
docker-compose restart       # Restart
```

## 🔧 Which Setup to Choose?

### Docker (Option 1) ✅
- ✅ **Zero configuration** - Everything pre-setup
- ✅ **No Python version conflicts**
- ✅ **No dependency installation issues**
- ✅ **Consistent across all systems**
- ✅ **All 43 API keys pre-configured**

### Python Virtual Environment (Option 2) 🐍
- ✅ **Direct access to code**
- ✅ **Faster development iteration**
- ✅ **Easy debugging**
- ✅ **No Docker requirement**
- ✅ **Full control over environment**

## ✨ Features

- ✅ **43 API Keys** - Automatic load balancing and failover
- ✅ **Zero Configuration** - Everything pre-configured
- ✅ **Document Processing** - PDF, DOCX, TXT support
- ✅ **Vector Search** - ChromaDB for fast retrieval
- ✅ **Multilingual** - Supports multiple languages
- ✅ **Persistent Data** - Vector database survives restarts

## 🏗️ Architecture

1. **Document Processing** → 2. **Embedding Generation** → 3. **Vector Storage** → 4. **Query Matching** → 5. **LLM Response**

## 🛠️ Tech Stack

- **FastAPI** - High-performance web framework
- **ChromaDB** - Vector storage and similarity search
- **Google Gemini** - Embeddings and LLM inference (43 API keys)
- **LangChain** - Document processing and chunking
- **Docker** - Containerization and deployment

## 📚 More Info

- **API Documentation**: http://localhost:8080/docs
- **Alternative Docs**: http://localhost:8080/redoc
- **Health Check**: http://localhost:8080/health
