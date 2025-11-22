"""
conftest.py

Shared pytest fixtures and configuration for Legal Risk Analysis System tests.
"""

import pytest
import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock deepagents before any imports
mock_deepagents = MagicMock()
mock_deepagents.create_deep_agent = MagicMock(return_value=MagicMock())
sys.modules['deepagents'] = mock_deepagents

# Mock web_research_tools (imported by agent_configuration.py but tools are in storage_and_tools)
mock_web_research = MagicMock()
mock_web_search = MagicMock()
mock_web_search.name = 'web_search'
mock_web_fetch = MagicMock()
mock_web_fetch.name = 'web_fetch'
mock_web_research.web_search = mock_web_search
mock_web_research.web_fetch = mock_web_fetch
sys.modules['web_research_tools'] = mock_web_research


def pytest_configure(config):
    """Create test fixtures before importing modules that need them."""
    import tempfile
    import json

    # Create a temporary data room for module-level imports
    temp_dir = tempfile.mkdtemp()
    index_path = os.path.join(temp_dir, "data_room_index.json")

    test_index = {
        "metadata": {"total_documents": 0, "total_pages": 0, "total_tokens": 0},
        "documents": []
    }

    with open(index_path, 'w') as f:
        json.dump(test_index, f)

    # Store for cleanup
    config._temp_data_dir = temp_dir

    # Patch the storage initialization paths before imports happen
    import builtins
    original_open = builtins.open

    def patched_open(file, *args, **kwargs):
        # Redirect storage_and_tools storage initialization to our temp file
        if 'preprocessed_data_room/data_room_index.json' in str(file):
            return original_open(index_path, *args, **kwargs)
        return original_open(file, *args, **kwargs)

    builtins.open = patched_open


def pytest_unconfigure(config):
    """Clean up temporary files after tests."""
    import shutil
    if hasattr(config, '_temp_data_dir'):
        shutil.rmtree(config._temp_data_dir, ignore_errors=True)


@pytest.fixture
def sample_data_room_index():
    """Create a sample data room index structure for testing."""
    return {
        "metadata": {
            "total_documents": 3,
            "total_pages": 15,
            "total_tokens": 5000
        },
        "documents": [
            {
                "document_id": "doc_001",
                "title": "Articles of Incorporation",
                "document_type": "Corporate Governance",
                "summary_description": "Corporate formation documents for TechCorp Inc.",
                "page_count": 5,
                "pages": [
                    {
                        "page_number": 1,
                        "summary_description": "Cover page with company name and state of incorporation",
                        "image_path": "page_images/doc_001/page_001.png",
                        "tokens_used": 100
                    },
                    {
                        "page_number": 2,
                        "summary_description": "Purpose clause and authorized shares",
                        "image_path": "page_images/doc_001/page_002.png",
                        "tokens_used": 120
                    },
                    {
                        "page_number": 3,
                        "summary_description": "Board composition and voting rights",
                        "image_path": "page_images/doc_001/page_003.png",
                        "tokens_used": 150
                    },
                    {
                        "page_number": 4,
                        "summary_description": "Amendment provisions",
                        "image_path": "page_images/doc_001/page_004.png",
                        "tokens_used": 110
                    },
                    {
                        "page_number": 5,
                        "summary_description": "Signature page",
                        "image_path": "page_images/doc_001/page_005.png",
                        "tokens_used": 80
                    }
                ],
                "pdf_path": "./data_room_pdfs/Articles_of_Incorporation.pdf",
                "total_tokens": 560
            },
            {
                "document_id": "doc_002",
                "title": "Master Services Agreement",
                "document_type": "Contracts",
                "summary_description": "MSA between TechCorp and Enterprise Client dated 2023",
                "page_count": 7,
                "pages": [
                    {
                        "page_number": 1,
                        "summary_description": "Title page and parties",
                        "image_path": "page_images/doc_002/page_001.png",
                        "tokens_used": 90
                    },
                    {
                        "page_number": 2,
                        "summary_description": "Definitions section",
                        "image_path": "page_images/doc_002/page_002.png",
                        "tokens_used": 200
                    },
                    {
                        "page_number": 3,
                        "summary_description": "Service obligations",
                        "image_path": "page_images/doc_002/page_003.png",
                        "tokens_used": 180
                    },
                    {
                        "page_number": 4,
                        "summary_description": "Payment terms",
                        "image_path": "page_images/doc_002/page_004.png",
                        "tokens_used": 160
                    },
                    {
                        "page_number": 5,
                        "summary_description": "Liability and indemnification",
                        "image_path": "page_images/doc_002/page_005.png",
                        "tokens_used": 220
                    },
                    {
                        "page_number": 6,
                        "summary_description": "Termination provisions",
                        "image_path": "page_images/doc_002/page_006.png",
                        "tokens_used": 170
                    },
                    {
                        "page_number": 7,
                        "summary_description": "Signature page with execution date",
                        "image_path": "page_images/doc_002/page_007.png",
                        "tokens_used": 100
                    }
                ],
                "pdf_path": "./data_room_pdfs/Master_Services_Agreement.pdf",
                "total_tokens": 1120
            },
            {
                "document_id": "doc_003",
                "title": "Patent Portfolio",
                "document_type": "Intellectual Property",
                "summary_description": "Summary of company's patent holdings",
                "page_count": 3,
                "pages": [
                    {
                        "page_number": 1,
                        "summary_description": "Patent inventory overview",
                        "image_path": "page_images/doc_003/page_001.png",
                        "tokens_used": 150
                    },
                    {
                        "page_number": 2,
                        "summary_description": "Patent details and expiration dates",
                        "image_path": "page_images/doc_003/page_002.png",
                        "tokens_used": 200
                    },
                    {
                        "page_number": 3,
                        "summary_description": "Licensing agreements summary",
                        "image_path": "page_images/doc_003/page_003.png",
                        "tokens_used": 180
                    }
                ],
                "pdf_path": "./data_room_pdfs/Patent_Portfolio.pdf",
                "total_tokens": 530
            }
        ]
    }


@pytest.fixture
def temp_data_room(sample_data_room_index, tmp_path):
    """Create a temporary data room directory with index and sample images."""
    # Create directory structure
    data_room_dir = tmp_path / "preprocessed_data_room"
    data_room_dir.mkdir()

    # Create page_images directories
    images_dir = data_room_dir / "page_images"
    images_dir.mkdir()

    # Create sample PNG files for each document
    for doc in sample_data_room_index["documents"]:
        doc_images_dir = images_dir / doc["document_id"]
        doc_images_dir.mkdir()

        for page in doc["pages"]:
            # Create a simple PNG file (1x1 pixel)
            image_path = data_room_dir / page["image_path"]
            # Simple 1x1 red PNG
            png_data = bytes([
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
                0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 dimensions
                0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,  # 8-bit RGB
                0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
                0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,  # compressed data
                0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
                0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,  # IEND chunk
                0x44, 0xAE, 0x42, 0x60, 0x82
            ])
            with open(image_path, 'wb') as f:
                f.write(png_data)

    # Save the index file
    index_path = data_room_dir / "data_room_index.json"
    with open(index_path, 'w') as f:
        json.dump(sample_data_room_index, f)

    return {
        "data_room_dir": data_room_dir,
        "index_path": index_path,
        "images_dir": images_dir
    }


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing."""
    mock_client = MagicMock()

    # Mock response for vision model
    mock_response = MagicMock()
    mock_response.output_text = "This is a sample page summary describing legal content."
    mock_client.responses.create.return_value = mock_response

    return mock_client


@pytest.fixture
def sample_page_image():
    """Create a sample page image bytes for testing."""
    # Simple 100x100 PNG image
    from io import BytesIO
    from PIL import Image

    img = Image.new('RGB', (100, 100), color='white')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def sample_interrupt_result():
    """Create a sample agent result with interrupt for testing."""
    class InterruptValue:
        def __init__(self):
            self.value = {
                "action_requests": [
                    {
                        "name": "get_documents",
                        "args": {
                            "document_ids": ["doc_001", "doc_002"]
                        }
                    }
                ],
                "review_configs": [
                    {
                        "action_name": "get_documents",
                        "allowed_decisions": ["approve", "reject", "edit"]
                    }
                ]
            }

    return {
        "__interrupt__": [InterruptValue()]
    }


@pytest.fixture
def sample_agent_messages():
    """Create sample agent messages for testing."""
    return {
        "messages": [
            {"role": "user", "content": "Analyze the data room"},
            {"role": "assistant", "content": "I will analyze the data room..."}
        ]
    }


@pytest.fixture
def mock_requests():
    """Create a mock for requests library."""
    with patch('requests.post') as mock_post, patch('requests.get') as mock_get:
        # Mock search response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "results": [
                {
                    "title": "Delaware Corporate Law",
                    "url": "https://example.com/law",
                    "snippet": "Information about Delaware corporate law...",
                    "domain": "example.com"
                }
            ]
        }

        # Mock fetch response
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"<html><body><p>Legal content here</p></body></html>"

        yield {"post": mock_post, "get": mock_get}


@pytest.fixture
def reset_global_counters():
    """Reset global usage counters before each test."""
    import storage_and_tools

    # Store original values
    original_page_count = storage_and_tools._page_retrieval_count
    original_web_count = storage_and_tools._web_fetch_count

    # Reset counters
    storage_and_tools._page_retrieval_count = 0
    storage_and_tools._web_fetch_count = 0

    yield

    # Restore original values
    storage_and_tools._page_retrieval_count = original_page_count
    storage_and_tools._web_fetch_count = original_web_count
