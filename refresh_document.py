#!/usr/bin/env python3
"""
Document Refresh Utility

A comprehensive tool for managing cached documents and vector store data.
Provides commands to list, remove, and clear documents with improved error handling
and user experience.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

try:
    from app.document_processor import get_url_hash, load_url_cache, save_url_cache
    from app.vector_store import init_vectorstore
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    print("Required modules: app.vector_store, app.document_processor")
    sys.exit(1)


class DocumentManager:
    """Manages document cache and vector store operations"""

    def __init__(self):
        self.cache_file = Path("./data/url_cache.json")
        self.vector_db_path = Path("./data/chroma_db")
        self.vectorstore = None

    def _get_vectorstore(self):
        """Lazy initialization of vectorstore"""
        if self.vectorstore is None:
            try:
                self.vectorstore = init_vectorstore()
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize vector store: {e}")
        return self.vectorstore

    def remove_from_vectorstore(self, url_hash: str) -> int:
        """Remove all chunks with specific url_hash from vector store"""
        try:
            vectorstore = self._get_vectorstore()
            if not vectorstore:
                return 0

            # Get underlying ChromaDB collection
            collection = vectorstore._collection

            # Get all documents with metadata
            results = collection.get(include=["metadatas", "documents"])

            # Find IDs that match our url_hash
            ids_to_delete = []
            metadatas = results.get("metadatas", []) or []
            ids = results.get("ids", []) or []

            for i, metadata in enumerate(metadatas):
                if metadata and metadata.get("url_hash") == url_hash and i < len(ids):
                    ids_to_delete.append(ids[i])

            # Delete the matching documents
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                print(f"   🗑️  Removed {len(ids_to_delete)} chunks from vector store")
            else:
                print("   i  No chunks found in vector store for this URL")

            return len(ids_to_delete)

        except Exception as e:
            print(f"   ⚠️  Error removing from vector store: {e}")
            return 0

    def remove_from_cache(self, url_hash: str) -> bool:
        """Remove document from cache"""
        try:
            cache = load_url_cache()
            if url_hash in cache:
                del cache[url_hash]
                save_url_cache(cache)
                print("   ✅ Removed from cache")
                return True
            else:
                print("   i  Not found in cache")
                return False
        except Exception as e:
            print(f"   ⚠️  Error removing from cache: {e}")
            return False

    def clear_document_completely(self, url: str) -> bool:
        """Clear document from both cache AND vector store"""
        if not url.strip():
            print("❌ Empty URL provided")
            return False

        url_hash = get_url_hash(url)

        print("🧹 Completely removing document:")
        print(f"   URL: {url[:80]}{'...' if len(url) > 80 else ''}")
        print(f"   Hash: {url_hash}")

        # Step 1: Remove from cache
        cache_removed = self.remove_from_cache(url_hash)

        # Step 2: Remove from vector store
        chunks_removed = self.remove_from_vectorstore(url_hash)

        success = cache_removed or chunks_removed > 0

        if success:
            print(
                f"   🎯 Cleanup summary: Cache {'✓' if cache_removed else '✗'}, Vector chunks: {chunks_removed}"
            )
            print("   💡 Document will be completely reprocessed on next run!")
        else:
            print("   ❌ Document not found in either cache or vector store")
            print("   💡 URL will be processed normally next time.")

        return success

    def list_cached_documents(self, show_details: bool = False) -> None:
        """List all documents currently in cache"""
        try:
            cache = load_url_cache()
            print(f"📄 Found {len(cache)} cached documents:")

            if not cache:
                print("   (No documents in cache)")
                return

            print("-" * 100)

            # Group by status for better readability
            processed = []
            pending = []

            for url_hash, data in cache.items():
                entry = {
                    "hash": url_hash,
                    "status": data
                    if isinstance(data, bool)
                    else data.get("processed", False),
                    "data": data,
                }

                if entry["status"]:
                    processed.append(entry)
                else:
                    pending.append(entry)

            # Display processed documents
            if processed:
                print(f"✅ Processed ({len(processed)}):")
                for entry in processed:
                    self._print_cache_entry(entry, show_details)

            # Display pending documents
            if pending:
                print(f"\n⏳ Pending ({len(pending)}):")
                for entry in pending:
                    self._print_cache_entry(entry, show_details)

        except Exception as e:
            print(f"❌ Error loading cache: {e}")

    def _print_cache_entry(self, entry: dict, show_details: bool) -> None:
        """Print a single cache entry"""
        hash_short = entry["hash"][:12] + "..."
        print(f"   {hash_short}")

        if show_details and isinstance(entry["data"], dict):
            data = entry["data"]
            if "url" in data:
                print(f"      URL: {data['url']}")
            if "timestamp" in data:
                print(f"      Time: {data['timestamp']}")
            if "chunks" in data:
                print(f"      Chunks: {data['chunks']}")

    def clear_all_data(self, force: bool = False) -> bool:
        """Clear entire cache AND vector store"""
        if not force:
            print("⚠️  This will remove ALL cached documents and vector data!")
            print("   All documents will need to be reprocessed from scratch.")
            confirm = input("   Continue? (yes/no): ").strip().lower()
            if confirm not in ["yes", "y"]:
                print("Operation cancelled.")
                return False

        print("🧹 Clearing ALL data (cache + vector store)...")

        cache_cleared = False
        vector_cleared = False

        # Clear cache
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                print("   ✅ Cleared entire document cache")
                cache_cleared = True
            else:
                print("   i  No cache file found")
        except Exception as e:
            print(f"   ⚠️  Error clearing cache: {e}")

        # Clear vector store
        try:
            if self.vector_db_path.exists():
                shutil.rmtree(self.vector_db_path)
                print("   ✅ Cleared entire vector database")
                vector_cleared = True
            else:
                print("   i  No vector database found")
        except Exception as e:
            print(f"   ⚠️  Error clearing vector database: {e}")

        if cache_cleared or vector_cleared:
            print("   🎯 Complete cleanup finished!")
            print("   💡 All documents will be completely reprocessed on next run!")
            return True
        else:
            print("   i  No data found to clear")
            return False

    def get_stats(self) -> dict:
        """Get statistics about cached documents and vector store"""
        stats = {
            "cache_exists": self.cache_file.exists(),
            "vector_db_exists": self.vector_db_path.exists(),
            "cached_documents": 0,
            "processed_documents": 0,
            "pending_documents": 0,
        }

        try:
            if stats["cache_exists"]:
                cache = load_url_cache()
                stats["cached_documents"] = len(cache)

                for data in cache.values():
                    if isinstance(data, bool):
                        if data:
                            stats["processed_documents"] += 1
                        else:
                            stats["pending_documents"] += 1
                    elif isinstance(data, dict) and data.get("processed", False):
                        stats["processed_documents"] += 1
                    else:
                        stats["pending_documents"] += 1
        except Exception as e:
            print(f"⚠️  Error getting cache stats: {e}")

        return stats


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Document Refresh Utility - Manage cached documents and vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                              # List cached documents
  %(prog)s list --details                    # List with detailed information
  %(prog)s remove https://example.com/doc    # Remove specific document
  %(prog)s clear-all                         # Clear everything (with confirmation)
  %(prog)s clear-all --force                 # Clear everything (no confirmation)
  %(prog)s stats                             # Show statistics
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List cached documents")
    list_parser.add_argument(
        "--details", action="store_true", help="Show detailed information"
    )

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove specific document")
    remove_parser.add_argument("url", help="URL of document to remove")

    # Clear-all command
    clear_parser = subparsers.add_parser("clear-all", help="Clear all data")
    clear_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    return parser


def main():
    """Main entry point"""
    parser = create_parser()

    # Handle no arguments
    if len(sys.argv) == 1:
        print("🔄 Document Refresh Utility")
        print("=" * 50)
        parser.print_help()
        return

    args = parser.parse_args()
    manager = DocumentManager()

    try:
        if args.command == "list":
            manager.list_cached_documents(show_details=args.details)

        elif args.command == "remove":
            if not args.url:
                print("❌ URL is required for remove command")
                return
            manager.clear_document_completely(args.url)

        elif args.command == "clear-all":
            manager.clear_all_data(force=args.force)

        elif args.command == "stats":
            stats = manager.get_stats()
            print("📊 Document Manager Statistics")
            print("-" * 40)
            print(f"Cache file exists: {'✅' if stats['cache_exists'] else '❌'}")
            print(f"Vector DB exists: {'✅' if stats['vector_db_exists'] else '❌'}")
            print(f"Total cached documents: {stats['cached_documents']}")
            print(f"  - Processed: {stats['processed_documents']}")
            print(f"  - Pending: {stats['pending_documents']}")

        else:
            print("❌ Unknown command")
            parser.print_help()

    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check your configuration and try again")


if __name__ == "__main__":
    main()
