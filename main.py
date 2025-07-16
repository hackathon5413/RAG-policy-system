#!/usr/bin/env python3
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.embedding_engine import EmbeddingEngine, PolicyQueryEngine

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [args]")
        print("Commands:")
        print("  process    - Process all PDFs and create embeddings")
        print("  search     - Search documents (requires query)")
        print("  stats      - Show system statistics")
        print("  query      - Process insurance query with decision logic")
        return
    
    command = sys.argv[1]
    
    if command == "process":
        print("🚀 Starting document processing...")
        engine = EmbeddingEngine()
        results = engine.process_all_documents()
        
        print(f"\n📊 Processing Results:")
        print(f"✅ Files processed: {len(results['processed_files'])}")
        print(f"📝 Total chunks created: {results['total_chunks']}")
        print(f"⏱️  Processing time: {results['processing_time']:.2f}s")
        
        if results['errors']:
            print(f"❌ Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"   {error}")
        
        engine.close()
    
    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: python main.py search <query>")
            return
        
        query = " ".join(sys.argv[2:])
        engine = EmbeddingEngine()
        
        print(f"🔍 Searching for: '{query}'")
        results = engine.search_documents(query, top_k=3)
        
        print(f"\n📋 Found {len(results['results'])} results:")
        for i, result in enumerate(results['results'], 1):
            print(f"\n{i}. {result['metadata']['filename']} (Page {result['metadata']['page']})")
            print(f"   Section: {result['metadata']['section_type']}")
            print(f"   Similarity: {result['similarity']:.3f}")
            print(f"   Content: {result['content'][:200]}...")
        
        engine.close()
    
    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python main.py query <insurance_query>")
            print("Example: python main.py query '46M knee surgery Pune 3-month policy'")
            return
        
        query = " ".join(sys.argv[2:])
        engine = PolicyQueryEngine()
        
        print(f"🏥 Processing insurance query: '{query}'")
        result = engine.process_insurance_query(query)
        
        print(f"\n📋 Decision: {result['decision']}")
        print(f"🎯 Confidence: {result['confidence']:.2f}")
        print(f"💭 Reasoning: {result['justification']['reasoning']}")
        
        print(f"\n📚 Relevant sections found:")
        for section in result['justification']['relevant_sections']:
            print(f"  • {section['source']} ({section['section_type']})")
            print(f"    Similarity: {section['similarity']:.3f}")
            print(f"    Content: {section['content'][:150]}...")
        
        engine.close()
    
    elif command == "stats":
        engine = EmbeddingEngine()
        stats = engine.get_system_stats()
        
        print("📊 System Statistics:")
        print(f"  Vector count: {stats['vector_count']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Files processed: {stats['total_files']}")
        print(f"  Section distribution:")
        for section_type, count in stats['section_distribution'].items():
            print(f"    {section_type}: {count}")
        
        engine.close()
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
