# Ollama Setup Instructions

## Quick Setup (5 minutes)

### 1. Run the setup script:
```bash
chmod +x setup_ollama.sh
./setup_ollama.sh
```

### 2. Manual setup (if script fails):
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# In another terminal, pull the model
ollama pull llama3.2:3b
```

### 3. Verify installation:
```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"llama3.2:3b","prompt":"Hello","stream":false}'
```

## System Requirements
- **RAM Usage:** ~2GB for llama3.2:3b
- **Storage:** ~2GB model download
- **Perfect for:** M1 8GB RAM systems

## What Changes
- ✅ **Chunk analysis:** FREE local processing
- ✅ **Final answers:** Still uses Gemini (premium quality)
- ✅ **Fallback:** If Ollama fails, uses Gemini
- ✅ **Cost savings:** 90%+ reduction in API calls

## Logs You'll See
```
🔍 [CHUNK ANALYSIS] Processing chunk 1 with Ollama
✅ [OLLAMA LOCAL] Completed chunk 1 analysis
⚡ [CHUNK ANALYSIS] Parallel processing for 25 chunks via Ollama (16 workers)
🎉 [CHUNK ANALYSIS] Enhanced 25 chunks with LOCAL LLM analysis
```

## Alternative Models (if 3B too slow)
- `llama3.2:1b` - Faster, less RAM
- `qwen2.5:3b` - Alternative 3B model
- `phi3.5:3.8b` - Microsoft's model

Change model in code: `call_ollama(prompt, model="llama3.2:1b")`
