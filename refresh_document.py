

import sys
import os

sys.path.append(os.path.dirname(__file__))

try:
    from app.vector_store import init_vectorstore
    from app.document_processor import get_url_hash, load_url_cache, save_url_cache
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def remove_document_from_vectorstore(url_hash: str) -> int:
    """Remove all chunks with specific url_hash from vector store"""
    try:
        vectorstore = init_vectorstore()
        
        # ChromaDB approach - get all documents and filter
        # Note: This is a workaround since ChromaDB doesn't have direct metadata-based deletion
        
        # Get underlying ChromaDB collection
        collection = vectorstore._collection
        
        # Get all documents with metadata
        results = collection.get(include=['metadatas', 'documents'])
        
        # Find IDs that match our url_hash
        ids_to_delete = []
        metadatas = results['metadatas'] or []
        for i, metadata in enumerate(metadatas):
            if metadata and metadata.get('url_hash') == url_hash:
                ids_to_delete.append(results['ids'][i])
        
        # Delete the matching documents
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"   🗑️  Removed {len(ids_to_delete)} chunks from vector store")
        else:
            print("   ℹ️  No chunks found in vector store for this URL")
            
        return len(ids_to_delete)
        
    except Exception as e:
        print(f"   ⚠️  Error removing from vector store: {e}")
        return 0

def clear_document_completely(url: str) -> bool:
    """Clear document from both cache AND vector store"""
    url_hash = get_url_hash(url)
    
    print(f"🧹 Completely removing document: {url[:60]}...")
    print(f"   Hash: {url_hash}")
    
    # Step 1: Remove from cache
    cache = load_url_cache()
    cache_removed = False
    if url_hash in cache:
        del cache[url_hash]
        save_url_cache(cache)
        print("   ✅ Removed from cache")
        cache_removed = True
    else:
        print("   ℹ️  Not found in cache")
    
    # Step 2: Remove from vector store
    chunks_removed = remove_document_from_vectorstore(url_hash)
    
    if cache_removed or chunks_removed > 0:
        print(f"   🎯 Total cleanup: Cache {'✓' if cache_removed else '✗'}, Vector chunks: {chunks_removed}")
        return True
    else:
        print("   ❌ Document not found in either cache or vector store")
        return False

def list_cached_documents():
    """List all documents currently in cache"""
    cache = load_url_cache()
    print(f"📄 Found {len(cache)} cached documents:")
    print("-" * 80)
    
    for url_hash, processed in cache.items():
        print(f"Hash: {url_hash} | Status: {'Processed' if processed else 'Pending'}")

def clear_all_data():
    """Clear entire cache AND vector store - everything will be reprocessed"""
    print("🧹 Clearing ALL data (cache + vector store)...")
    
    # Clear cache
    cache_file = "./data/url_cache.json"
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("   ✅ Cleared entire document cache")
    else:
        print("   ℹ️  No cache file found")
    
    # Clear vector store
    vector_db_path = "./data/chroma_db"
    if os.path.exists(vector_db_path):
        import shutil
        shutil.rmtree(vector_db_path)
        print("   ✅ Cleared entire vector database")
    else:
        print("   ℹ️  No vector database found")
    
    print("   🎯 Complete cleanup finished!")

def main():
    if len(sys.argv) < 2:
        print("🔄 Document Refresh Utility")
        print("=" * 50)
        print("Usage:")
        print("  python refresh_document.py list                    # List cached documents")
        print("  python refresh_document.py remove <URL>           # Remove document completely")
        print("  python refresh_document.py clear-all              # Clear everything")
        print("")
        print("Examples:")
        print("  python refresh_document.py remove https://example.com/document.pdf")
        print("  python refresh_document.py list")
        print("")
        print("📝 Note: 'remove' clears BOTH cache AND vector store for complete refresh")
        return

    command = sys.argv[1].lower()

    if command == "list":
        list_cached_documents()
    
    elif command in ["remove", "clear"] and len(sys.argv) > 2:
        url = sys.argv[2]
        if clear_document_completely(url):
            print("💡 Next time you process this URL, it will be completely reprocessed!")
        else:
            print("💡 URL was not found, but it will be processed normally next time.")
    
    elif command == "clear-all":
        confirm = input("⚠️  Are you sure you want to clear ALL data (cache + vector store)? (yes/no): ")
        if confirm.lower() == 'yes':
            clear_all_data()
            print("💡 All documents will be completely reprocessed on next run!")
        else:
            print("Operation cancelled.")
    
    else:
        print("❌ Invalid command. Use 'list', 'remove <URL>', or 'clear-all'")

if __name__ == "__main__":
    main()
