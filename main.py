#!/usr/bin/env python3

import sys
from rag_system import process_pdf, process_directory, search_documents, answer_question, get_stats, format_search_results


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [args]")
        print("\nCommands:")
        print("  ingest <pdf_path>     - Process single PDF file")
        print("  ingest-dir <dir_path> - Process all PDFs in directory")
        print("  query <question>      - Ask insurance question")
        print("  search <text>         - Search documents")
        print("  stats                 - Show system statistics")
        return
    
    command = sys.argv[1]
    
    if command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python main.py ingest <pdf_path>")
            return
        
        pdf_path = sys.argv[2]
        print(f"📄 Processing: {pdf_path}")
        
        result = process_pdf(pdf_path)
        if result["success"]:
            print(f"✅ Success: {result['chunks_created']} chunks from {result['pages_processed']} pages")
        else:
            print(f"❌ Error: {result['error']}")
    
    elif command == "ingest-dir":
        dir_path = sys.argv[2] if len(sys.argv) > 2 else "./assets"
        print(f"📁 Processing directory: {dir_path}")
        
        results = process_directory(dir_path)
        if results["processed_files"]:
            print(f"✅ Processed {len(results['processed_files'])} files")
            print(f"📝 Total chunks: {results['total_chunks']}")
        
        if results["errors"]:
            print(f"❌ Errors: {len(results['errors'])}")
            for error in results["errors"]:
                print(f"   {error}")
    
    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python main.py query '<question>'")
            print("Example: python main.py query 'Is maternity covered?'")
            return
        
        question = " ".join(sys.argv[2:])
        print(f"❓ Question: {question}")
        print("🤖 Generating answer...\n")
        
        result = answer_question(question)
        print(f"💬 Answer: {result['answer']}")
        
        if result['sources']:
            print("\n📚 Sources:")
            for source in result['sources']:
                print(f"  • {source}")
    
    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: python main.py search '<text>'")
            return
        
        search_text = " ".join(sys.argv[2:])
        print(f"🔍 Searching: {search_text}")
        
        results = search_documents(search_text)
        formatted_output = format_search_results(search_text, results)
        print(formatted_output)
    
    elif command == "stats":
        stats = get_stats()
        print("📊 System Statistics:")
        if "error" in stats:
            print(f"❌ Error: {stats['error']}")
        else:
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Embedding model: {stats['embedding_model']}")
            print(f"  Embedding dimensions: {stats['embedding_dimensions']}")
            print(f"  Chunk size: {stats['chunk_settings']['size']}")
            print(f"  Chunk overlap: {stats['chunk_settings']['overlap']}")
    
    else:
        print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    main()
