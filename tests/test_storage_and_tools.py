"""
test_storage_and_tools.py

Tests for the storage layer and data room access tools.
"""

import pytest
import json
import base64
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataRoomStorage:
    """Tests for DataRoomStorage class."""

    def test_storage_initialization(self, temp_data_room):
        """Test successful storage initialization."""
        from storage_and_tools import DataRoomStorage

        storage = DataRoomStorage(
            str(temp_data_room["index_path"]),
            str(temp_data_room["data_room_dir"])
        )

        assert len(storage.documents) == 3
        assert storage.metadata["total_documents"] == 3
        assert storage.metadata["total_pages"] == 15

    def test_storage_loads_documents_by_id(self, temp_data_room):
        """Test that documents are indexed by ID."""
        from storage_and_tools import DataRoomStorage

        storage = DataRoomStorage(
            str(temp_data_room["index_path"]),
            str(temp_data_room["data_room_dir"])
        )

        assert "doc_001" in storage.documents
        assert "doc_002" in storage.documents
        assert "doc_003" in storage.documents
        assert storage.documents["doc_001"]["title"] == "Articles of Incorporation"

    def test_list_all_documents(self, temp_data_room):
        """Test listing all documents."""
        from storage_and_tools import DataRoomStorage

        storage = DataRoomStorage(
            str(temp_data_room["index_path"]),
            str(temp_data_room["data_room_dir"])
        )

        docs = storage.list_all_documents()

        assert len(docs) == 3
        assert docs[0]["id"] == "doc_001"
        assert docs[0]["title"] == "Articles of Incorporation"
        assert docs[0]["document_type"] == "Corporate Governance"
        assert docs[0]["page_count"] == 5
        assert "summary_description" in docs[0]

    def test_get_document_page_summaries(self, temp_data_room):
        """Test retrieving page summaries for documents."""
        from storage_and_tools import DataRoomStorage

        storage = DataRoomStorage(
            str(temp_data_room["index_path"]),
            str(temp_data_room["data_room_dir"])
        )

        result = storage.get_document_page_summaries(["doc_001"])

        assert "doc_001" in result
        assert result["doc_001"]["title"] == "Articles of Incorporation"
        assert len(result["doc_001"]["pages"]) == 5
        assert result["doc_001"]["pages"][0]["page_number"] == 1

    def test_get_multiple_document_summaries(self, temp_data_room):
        """Test retrieving summaries for multiple documents."""
        from storage_and_tools import DataRoomStorage

        storage = DataRoomStorage(
            str(temp_data_room["index_path"]),
            str(temp_data_room["data_room_dir"])
        )

        result = storage.get_document_page_summaries(["doc_001", "doc_002"])

        assert len(result) == 2
        assert "doc_001" in result
        assert "doc_002" in result

    def test_get_document_page_summaries_invalid_id(self, temp_data_room):
        """Test handling of invalid document ID."""
        from storage_and_tools import DataRoomStorage

        storage = DataRoomStorage(
            str(temp_data_room["index_path"]),
            str(temp_data_room["data_room_dir"])
        )

        result = storage.get_document_page_summaries(["invalid_doc"])

        assert result == {}

    def test_get_page_images(self, temp_data_room):
        """Test retrieving page images."""
        from storage_and_tools import DataRoomStorage

        storage = DataRoomStorage(
            str(temp_data_room["index_path"]),
            str(temp_data_room["data_room_dir"])
        )

        result = storage.get_page_images("doc_001", [1, 2])

        assert result["document_id"] == "doc_001"
        assert result["title"] == "Articles of Incorporation"
        assert len(result["pages"]) == 2
        assert "image_data" in result["pages"][0]
        assert result["pages"][0]["image_data"].startswith("data:image/png;base64,")

    def test_get_page_images_invalid_document(self, temp_data_room):
        """Test get_page_images with invalid document ID."""
        from storage_and_tools import DataRoomStorage

        storage = DataRoomStorage(
            str(temp_data_room["index_path"]),
            str(temp_data_room["data_room_dir"])
        )

        result = storage.get_page_images("invalid_doc", [1])

        assert "error" in result
        assert "not found" in result["error"]

    def test_get_page_images_invalid_page(self, temp_data_room):
        """Test get_page_images with invalid page number."""
        from storage_and_tools import DataRoomStorage

        storage = DataRoomStorage(
            str(temp_data_room["index_path"]),
            str(temp_data_room["data_room_dir"])
        )

        result = storage.get_page_images("doc_001", [999])

        assert len(result["pages"]) == 1
        assert "error" in result["pages"][0]

    def test_get_page_images_missing_file(self, temp_data_room, sample_data_room_index):
        """Test get_page_images when image file is missing."""
        from storage_and_tools import DataRoomStorage
        import os

        # Remove one of the image files
        image_path = temp_data_room["data_room_dir"] / "page_images" / "doc_001" / "page_001.png"
        os.remove(image_path)

        storage = DataRoomStorage(
            str(temp_data_room["index_path"]),
            str(temp_data_room["data_room_dir"])
        )

        result = storage.get_page_images("doc_001", [1])

        assert "error" in result["pages"][0]
        assert "not found" in result["pages"][0]["error"]


class TestListDataRoomDocumentsTool:
    """Tests for list_data_room_documents tool."""

    @patch('storage_and_tools.data_room_storage')
    def test_list_documents_success(self, mock_storage):
        """Test successful document listing."""
        from storage_and_tools import list_data_room_documents

        mock_storage.list_all_documents.return_value = [
            {
                "id": "doc_001",
                "title": "Test Doc",
                "summary_description": "A test document",
                "document_type": "Contract",
                "page_count": 5
            }
        ]

        result = list_data_room_documents.invoke({})

        assert "doc_001" in result
        assert "Test Doc" in result
        assert "Contract" in result

    @patch('storage_and_tools.data_room_storage')
    def test_list_documents_empty(self, mock_storage):
        """Test listing when no documents exist."""
        from storage_and_tools import list_data_room_documents

        mock_storage.list_all_documents.return_value = []

        result = list_data_room_documents.invoke({})

        assert "No documents found" in result


class TestGetDocumentsTool:
    """Tests for get_documents tool."""

    @patch('storage_and_tools.data_room_storage')
    def test_get_documents_success(self, mock_storage):
        """Test successful document retrieval."""
        from storage_and_tools import get_documents

        mock_storage.get_document_page_summaries.return_value = {
            "doc_001": {
                "title": "Test Doc",
                "document_type": "Contract",
                "summary": "Summary text",
                "pages": [
                    {"page_number": 1, "summary": "Page 1 summary"}
                ]
            }
        }

        result = get_documents.invoke({"document_ids": ["doc_001"]})

        assert "Test Doc" in result
        assert "Page 1" in result

    @patch('storage_and_tools.data_room_storage')
    def test_get_documents_empty_list(self, mock_storage):
        """Test get_documents with empty document list."""
        from storage_and_tools import get_documents

        result = get_documents.invoke({"document_ids": []})

        assert "Error" in result

    @patch('storage_and_tools.data_room_storage')
    def test_get_documents_not_found(self, mock_storage):
        """Test get_documents when documents not found."""
        from storage_and_tools import get_documents

        mock_storage.get_document_page_summaries.return_value = {}

        result = get_documents.invoke({"document_ids": ["invalid"]})

        assert "Error" in result


class TestGetDocumentPagesTool:
    """Tests for get_document_pages tool with usage limits."""

    @patch('storage_and_tools.data_room_storage')
    def test_get_pages_success(self, mock_storage):
        """Test successful page retrieval."""
        import storage_and_tools
        storage_and_tools._page_retrieval_count = 0

        mock_storage.get_page_images.return_value = {
            "document_id": "doc_001",
            "title": "Test Doc",
            "pages": [
                {
                    "page_number": 1,
                    "summary": "Page summary",
                    "image_data": "data:image/png;base64,..."
                }
            ]
        }

        from storage_and_tools import get_document_pages
        result = get_document_pages.invoke({
            "document_id": "doc_001",
            "page_numbers": [1]
        })

        assert "Retrieved 1 pages" in result
        assert storage_and_tools._page_retrieval_count == 1

    @patch('storage_and_tools.data_room_storage')
    def test_get_pages_empty_list(self, mock_storage):
        """Test get_document_pages with empty page list."""
        from storage_and_tools import get_document_pages

        result = get_document_pages.invoke({
            "document_id": "doc_001",
            "page_numbers": []
        })

        assert "Error" in result

    @patch('storage_and_tools.data_room_storage')
    def test_get_pages_limit_reached(self, mock_storage):
        """Test get_document_pages when limit is reached."""
        import storage_and_tools
        storage_and_tools._page_retrieval_count = 50  # At limit

        from storage_and_tools import get_document_pages
        result = get_document_pages.invoke({
            "document_id": "doc_001",
            "page_numbers": [1]
        })

        assert "limit reached" in result

        # Reset for other tests
        storage_and_tools._page_retrieval_count = 0

    @patch('storage_and_tools.data_room_storage')
    def test_get_pages_would_exceed_limit(self, mock_storage):
        """Test get_document_pages when request would exceed limit."""
        import storage_and_tools
        storage_and_tools._page_retrieval_count = 48  # 2 remaining

        from storage_and_tools import get_document_pages
        result = get_document_pages.invoke({
            "document_id": "doc_001",
            "page_numbers": [1, 2, 3, 4, 5]  # Request 5
        })

        assert "exceed" in result
        assert "2" in result  # Remaining quota

        # Reset
        storage_and_tools._page_retrieval_count = 0

    @patch('storage_and_tools.data_room_storage')
    def test_get_pages_document_not_found(self, mock_storage):
        """Test get_document_pages when document not found."""
        import storage_and_tools
        storage_and_tools._page_retrieval_count = 0

        mock_storage.get_page_images.return_value = {
            "error": "Document not found"
        }

        from storage_and_tools import get_document_pages
        result = get_document_pages.invoke({
            "document_id": "invalid",
            "page_numbers": [1]
        })

        assert "Error" in result

    @patch('storage_and_tools.data_room_storage')
    def test_get_pages_updates_counter(self, mock_storage):
        """Test that successful retrieval updates the counter."""
        import storage_and_tools
        storage_and_tools._page_retrieval_count = 0

        mock_storage.get_page_images.return_value = {
            "document_id": "doc_001",
            "title": "Test Doc",
            "pages": [
                {"page_number": 1, "summary": "S1", "image_data": "..."},
                {"page_number": 2, "summary": "S2", "image_data": "..."},
                {"page_number": 3, "summary": "S3", "image_data": "..."}
            ]
        }

        from storage_and_tools import get_document_pages
        get_document_pages.invoke({
            "document_id": "doc_001",
            "page_numbers": [1, 2, 3]
        })

        assert storage_and_tools._page_retrieval_count == 3

        # Reset
        storage_and_tools._page_retrieval_count = 0


class TestWebSearchTool:
    """Tests for web_search tool."""

    @patch('storage_and_tools.requests.post')
    def test_web_search_success(self, mock_post):
        """Test successful web search."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "results": [
                {
                    "title": "Delaware Corporate Law",
                    "url": "https://example.com/law",
                    "snippet": "Information about corporate law",
                    "domain": "example.com"
                }
            ]
        }

        from storage_and_tools import web_search
        result = web_search.invoke({"query": "Delaware corporate law"})

        assert "Delaware Corporate Law" in result
        assert "example.com" in result

    @patch('storage_and_tools.requests.post')
    def test_web_search_empty_query(self, mock_post):
        """Test web_search with empty query."""
        from storage_and_tools import web_search
        result = web_search.invoke({"query": "   "})

        assert "Error" in result

    @patch('storage_and_tools.requests.post')
    def test_web_search_no_results(self, mock_post):
        """Test web_search with no results."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"results": []}

        from storage_and_tools import web_search
        result = web_search.invoke({"query": "nonexistent topic"})

        assert "No results" in result

    @patch('storage_and_tools.requests.post')
    def test_web_search_api_error(self, mock_post):
        """Test web_search handling of API error."""
        mock_post.return_value.status_code = 500

        from storage_and_tools import web_search
        result = web_search.invoke({"query": "test"})

        assert "Error" in result
        assert "500" in result

    @patch('storage_and_tools.requests.post')
    def test_web_search_timeout(self, mock_post):
        """Test web_search handling of timeout."""
        import requests
        mock_post.side_effect = requests.Timeout()

        from storage_and_tools import web_search
        result = web_search.invoke({"query": "test"})

        assert "timed out" in result

    @patch('storage_and_tools.requests.post')
    def test_web_search_custom_max_results(self, mock_post):
        """Test web_search with custom max_results."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"results": []}

        from storage_and_tools import web_search
        web_search.invoke({"query": "test", "max_results": 10})

        call_args = mock_post.call_args
        assert call_args.kwargs["json"]["max_results"] == 10


class TestWebFetchTool:
    """Tests for web_fetch tool with usage limits."""

    @patch('storage_and_tools.requests.get')
    def test_web_fetch_success(self, mock_get):
        """Test successful web fetch."""
        import storage_and_tools
        storage_and_tools._web_fetch_count = 0

        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"<html><body><p>Legal content</p></body></html>"

        from storage_and_tools import web_fetch
        result = web_fetch.invoke({"url": "https://example.com/page"})

        assert "Legal content" in result
        assert storage_and_tools._web_fetch_count == 1

        # Reset
        storage_and_tools._web_fetch_count = 0

    @patch('storage_and_tools.requests.get')
    def test_web_fetch_empty_url(self, mock_get):
        """Test web_fetch with empty URL."""
        from storage_and_tools import web_fetch
        result = web_fetch.invoke({"url": "   "})

        assert "Error" in result

    @patch('storage_and_tools.requests.get')
    def test_web_fetch_limit_reached(self, mock_get):
        """Test web_fetch when limit is reached."""
        import storage_and_tools
        storage_and_tools._web_fetch_count = 20  # At limit

        from storage_and_tools import web_fetch
        result = web_fetch.invoke({"url": "https://example.com"})

        assert "limit reached" in result

        # Reset
        storage_and_tools._web_fetch_count = 0

    @patch('storage_and_tools.requests.get')
    def test_web_fetch_http_error(self, mock_get):
        """Test web_fetch handling of HTTP error."""
        import storage_and_tools
        storage_and_tools._web_fetch_count = 0

        mock_get.return_value.status_code = 404

        from storage_and_tools import web_fetch
        result = web_fetch.invoke({"url": "https://example.com/notfound"})

        assert "Error" in result
        assert "404" in result

        # Verify counter not incremented on failure
        assert storage_and_tools._web_fetch_count == 0

    @patch('storage_and_tools.requests.get')
    def test_web_fetch_timeout(self, mock_get):
        """Test web_fetch handling of timeout."""
        import storage_and_tools
        storage_and_tools._web_fetch_count = 0
        import requests
        mock_get.side_effect = requests.Timeout()

        from storage_and_tools import web_fetch
        result = web_fetch.invoke({"url": "https://example.com"})

        assert "timed out" in result

    @patch('storage_and_tools.requests.get')
    def test_web_fetch_content_truncation(self, mock_get):
        """Test web_fetch truncates very long content."""
        import storage_and_tools
        storage_and_tools._web_fetch_count = 0

        # Create content longer than 8000 chars
        long_content = "<html><body>" + "x" * 10000 + "</body></html>"
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = long_content.encode()

        from storage_and_tools import web_fetch
        result = web_fetch.invoke({"url": "https://example.com"})

        assert "truncated" in result

        # Reset
        storage_and_tools._web_fetch_count = 0

    @patch('storage_and_tools.requests.get')
    def test_web_fetch_strips_scripts(self, mock_get):
        """Test web_fetch removes script tags."""
        import storage_and_tools
        storage_and_tools._web_fetch_count = 0

        html = """
        <html>
        <body>
        <script>alert('evil');</script>
        <p>Good content</p>
        <style>.hidden { display: none; }</style>
        </body>
        </html>
        """
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = html.encode()

        from storage_and_tools import web_fetch
        result = web_fetch.invoke({"url": "https://example.com"})

        assert "Good content" in result
        assert "alert" not in result

        # Reset
        storage_and_tools._web_fetch_count = 0

    @patch('storage_and_tools.requests.get')
    def test_web_fetch_shows_remaining_quota(self, mock_get):
        """Test web_fetch shows remaining quota."""
        import storage_and_tools
        storage_and_tools._web_fetch_count = 5

        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"<html><body>Content</body></html>"

        from storage_and_tools import web_fetch
        result = web_fetch.invoke({"url": "https://example.com"})

        assert "14" in result  # 20 - 5 - 1 = 14 remaining

        # Reset
        storage_and_tools._web_fetch_count = 0


class TestUsageLimitsIntegration:
    """Integration tests for usage limit behavior."""

    @patch('storage_and_tools.data_room_storage')
    def test_page_retrieval_limit_enforced_across_calls(self, mock_storage):
        """Test that page retrieval limit is enforced across multiple calls."""
        import storage_and_tools
        storage_and_tools._page_retrieval_count = 0

        mock_storage.get_page_images.return_value = {
            "document_id": "doc_001",
            "title": "Test",
            "pages": [{"page_number": 1, "summary": "S", "image_data": "..."}]
        }

        from storage_and_tools import get_document_pages

        # Make multiple calls
        for i in range(50):
            result = get_document_pages.invoke({
                "document_id": "doc_001",
                "page_numbers": [1]
            })

        # 51st call should fail
        result = get_document_pages.invoke({
            "document_id": "doc_001",
            "page_numbers": [1]
        })

        assert "limit reached" in result

        # Reset
        storage_and_tools._page_retrieval_count = 0

    @patch('storage_and_tools.requests.get')
    def test_web_fetch_limit_enforced_across_calls(self, mock_get):
        """Test that web fetch limit is enforced across multiple calls."""
        import storage_and_tools
        storage_and_tools._web_fetch_count = 0

        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"<html><body>Content</body></html>"

        from storage_and_tools import web_fetch

        # Make 20 calls (the limit)
        for i in range(20):
            result = web_fetch.invoke({"url": f"https://example.com/page{i}"})

        # 21st call should fail
        result = web_fetch.invoke({"url": "https://example.com/page21"})

        assert "limit reached" in result

        # Reset
        storage_and_tools._web_fetch_count = 0
