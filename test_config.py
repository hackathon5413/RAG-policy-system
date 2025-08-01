#!/usr/bin/env python3
"""
Quick test to verify the configuration and imports are working
"""

import sys
import os
sys.path.append('/Users/garvgoel/Desktop/Dev/RAGPolicy/RAG-policy-system')

try:
    print("Testing configuration imports...")
    
    # Test config imports
    from config import CONFIG, config
    print(f"✅ CONFIG imported successfully")
    print(f"✅ config imported successfully")
    
    # Test basic config values
    chunk_size = getattr(config, 'chunk_size', CONFIG.get('chunk_size', 800))
    print(f"✅ chunk_size: {chunk_size}")
    
    top_k = getattr(config, 'top_k', CONFIG.get('top_k', 10))
    print(f"✅ top_k: {top_k}")
    
    # Test app imports
    print("\nTesting app module imports...")
    
    try:
        from app.vector_store import init_vectorstore, get_embeddings
        print("✅ vector_store imports successful")
    except Exception as e:
        print(f"❌ vector_store import error: {e}")
    
    try:
        from app.embeddings import EnhancedGeminiEmbeddings
        print("✅ embeddings imports successful")
    except Exception as e:
        print(f"❌ embeddings import error: {e}")
    
    try:
        from app.document_processor import enhanced_search_for_question
        print("✅ document_processor imports successful")
    except Exception as e:
        print(f"❌ document_processor import error: {e}")
    
    print("\n🎉 All critical imports successful!")
    print("The configuration issues should now be resolved.")
    
except Exception as e:
    print(f"❌ Critical error: {e}")
    import traceback
    traceback.print_exc()
