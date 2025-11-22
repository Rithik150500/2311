"""
test_legal_preprocessing.py

Tests for the legal preprocessing module that converts PDFs to structured summaries.
"""

import pytest
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from io import BytesIO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from legal_preprocessing import (
    PageSummary,
    DocumentSummary,
    DataRoom,
    calculate_image_tokens,
    extract_pages_from_pdf,
    summarize_page_image,
    summarize_document_from_pages,
    process_data_room
)


class TestPageSummary:
    """Tests for PageSummary dataclass."""

    def test_page_summary_creation(self):
        """Test creating a PageSummary instance."""
        page = PageSummary(
            page_number=1,
            summary_description="Test summary",
            image_path="/path/to/image.png",
            tokens_used=100
        )
        assert page.page_number == 1
        assert page.summary_description == "Test summary"
        assert page.image_path == "/path/to/image.png"
        assert page.tokens_used == 100

    def test_page_summary_default_values(self):
        """Test PageSummary with various values."""
        page = PageSummary(
            page_number=0,
            summary_description="",
            image_path="",
            tokens_used=0
        )
        assert page.page_number == 0
        assert page.summary_description == ""


class TestDocumentSummary:
    """Tests for DocumentSummary dataclass."""

    def test_document_summary_creation(self):
        """Test creating a DocumentSummary instance."""
        doc = DocumentSummary(
            document_id="doc_001",
            title="Test Document",
            document_type="Contract",
            summary_description="A test document",
            page_count=5,
            pages=[],
            pdf_path="/path/to/doc.pdf",
            total_tokens=500
        )
        assert doc.document_id == "doc_001"
        assert doc.title == "Test Document"
        assert doc.document_type == "Contract"
        assert doc.page_count == 5
        assert doc.total_tokens == 500

    def test_document_summary_with_pages(self):
        """Test DocumentSummary with page data."""
        pages = [
            {"page_number": 1, "summary_description": "Page 1"},
            {"page_number": 2, "summary_description": "Page 2"}
        ]
        doc = DocumentSummary(
            document_id="doc_002",
            title="Multi-page Doc",
            document_type="Agreement",
            summary_description="Summary",
            page_count=2,
            pages=pages,
            pdf_path="/path/to/doc.pdf",
            total_tokens=200
        )
        assert len(doc.pages) == 2
        assert doc.pages[0]["page_number"] == 1


class TestDataRoom:
    """Tests for DataRoom dataclass and its methods."""

    def test_data_room_creation(self):
        """Test creating a DataRoom instance."""
        data_room = DataRoom(
            documents=[],
            total_documents=0,
            total_pages=0,
            total_tokens=0
        )
        assert data_room.total_documents == 0
        assert data_room.total_pages == 0
        assert data_room.total_tokens == 0

    def test_data_room_with_documents(self):
        """Test DataRoom with multiple documents."""
        docs = [
            {"document_id": "doc_001", "title": "Doc 1"},
            {"document_id": "doc_002", "title": "Doc 2"}
        ]
        data_room = DataRoom(
            documents=docs,
            total_documents=2,
            total_pages=10,
            total_tokens=1000
        )
        assert data_room.total_documents == 2
        assert len(data_room.documents) == 2

    def test_data_room_to_json(self, tmp_path):
        """Test saving DataRoom to JSON file."""
        docs = [
            {"document_id": "doc_001", "title": "Test Doc", "page_count": 5}
        ]
        data_room = DataRoom(
            documents=docs,
            total_documents=1,
            total_pages=5,
            total_tokens=500
        )

        output_path = tmp_path / "output" / "index.json"
        data_room.to_json(str(output_path))

        assert output_path.exists()

        with open(output_path, 'r') as f:
            saved_data = json.load(f)

        assert saved_data["metadata"]["total_documents"] == 1
        assert saved_data["metadata"]["total_pages"] == 5
        assert saved_data["metadata"]["total_tokens"] == 500
        assert len(saved_data["documents"]) == 1

    def test_data_room_to_json_creates_parent_dirs(self, tmp_path):
        """Test that to_json creates parent directories."""
        data_room = DataRoom(
            documents=[],
            total_documents=0,
            total_pages=0,
            total_tokens=0
        )

        nested_path = tmp_path / "deep" / "nested" / "path" / "index.json"
        data_room.to_json(str(nested_path))

        assert nested_path.exists()


class TestCalculateImageTokens:
    """Tests for calculate_image_tokens function."""

    def test_small_image_tokens(self):
        """Test token calculation for small image."""
        # 100x100 image: ceil(100/32) * ceil(100/32) = 4 * 4 = 16 patches
        # 16 * 1.62 = 25.92, ceil = 26
        tokens = calculate_image_tokens(100, 100)
        expected = math.ceil(16 * 1.62)
        assert tokens == expected

    def test_standard_image_tokens(self):
        """Test token calculation for standard document image."""
        # 1700x2200 image (typical letter size at 200 DPI)
        tokens = calculate_image_tokens(1700, 2200)
        # This exceeds 1536 patches, so scaling applies
        raw_patches = math.ceil(1700/32) * math.ceil(2200/32)
        assert raw_patches > 1536
        assert tokens > 0

    def test_exact_32x32_image(self):
        """Test token calculation for 32x32 image."""
        # 32x32 = exactly 1 patch
        tokens = calculate_image_tokens(32, 32)
        expected = math.ceil(1 * 1.62)
        assert tokens == expected

    def test_different_models(self):
        """Test token calculation with different model multipliers."""
        width, height = 64, 64
        # 2x2 = 4 patches

        mini_tokens = calculate_image_tokens(width, height, "gpt-4.1-mini")
        nano_tokens = calculate_image_tokens(width, height, "gpt-4.1-nano")

        # nano has higher multiplier (2.46 vs 1.62)
        assert nano_tokens > mini_tokens

    def test_unknown_model_uses_default(self):
        """Test that unknown model uses default multiplier."""
        tokens_unknown = calculate_image_tokens(100, 100, "unknown-model")
        tokens_default = calculate_image_tokens(100, 100, "gpt-4.1-mini")
        assert tokens_unknown == tokens_default

    def test_large_image_scaling(self):
        """Test that large images are scaled down properly."""
        # Very large image that exceeds 1536 patch limit
        tokens = calculate_image_tokens(10000, 10000)
        # Should be scaled down but still return reasonable value
        assert tokens > 0
        assert tokens < 10000  # Sanity check


class TestExtractPagesFromPdf:
    """Tests for extract_pages_from_pdf function."""

    @patch('legal_preprocessing.convert_from_path')
    def test_successful_extraction(self, mock_convert):
        """Test successful PDF page extraction."""
        # Create mock PIL images
        from PIL import Image
        mock_images = [
            Image.new('RGB', (100, 100), 'white'),
            Image.new('RGB', (100, 100), 'gray')
        ]
        mock_convert.return_value = mock_images

        result = extract_pages_from_pdf("/path/to/test.pdf")

        assert len(result) == 2
        assert isinstance(result[0], bytes)
        assert isinstance(result[1], bytes)
        mock_convert.assert_called_once_with("/path/to/test.pdf", dpi=200, fmt='png')

    @patch('legal_preprocessing.convert_from_path')
    def test_custom_dpi(self, mock_convert):
        """Test PDF extraction with custom DPI."""
        from PIL import Image
        mock_convert.return_value = [Image.new('RGB', (100, 100), 'white')]

        extract_pages_from_pdf("/path/to/test.pdf", dpi=300)

        mock_convert.assert_called_once_with("/path/to/test.pdf", dpi=300, fmt='png')

    @patch('legal_preprocessing.convert_from_path')
    def test_extraction_failure(self, mock_convert):
        """Test handling of extraction failure."""
        mock_convert.side_effect = Exception("PDF conversion failed")

        result = extract_pages_from_pdf("/path/to/bad.pdf")

        assert result == []

    @patch('legal_preprocessing.convert_from_path')
    def test_empty_pdf(self, mock_convert):
        """Test handling of empty PDF."""
        mock_convert.return_value = []

        result = extract_pages_from_pdf("/path/to/empty.pdf")

        assert result == []


class TestSummarizePageImage:
    """Tests for summarize_page_image function."""

    @patch('legal_preprocessing.client')
    def test_successful_summarization(self, mock_client, sample_page_image):
        """Test successful page summarization."""
        mock_response = MagicMock()
        mock_response.output_text = "This page contains legal definitions."
        mock_client.responses.create.return_value = mock_response

        summary, tokens = summarize_page_image(
            sample_page_image,
            page_number=1,
            document_name="Test Doc"
        )

        assert summary == "This page contains legal definitions."
        assert tokens > 0
        mock_client.responses.create.assert_called_once()

    @patch('legal_preprocessing.client')
    def test_summarization_with_custom_model(self, mock_client, sample_page_image):
        """Test summarization with custom model."""
        mock_response = MagicMock()
        mock_response.output_text = "Summary text"
        mock_client.responses.create.return_value = mock_response

        summarize_page_image(
            sample_page_image,
            page_number=1,
            document_name="Test",
            model="gpt-4.1-nano"
        )

        call_args = mock_client.responses.create.call_args
        assert call_args.kwargs['model'] == "gpt-4.1-nano"

    @patch('legal_preprocessing.client')
    def test_summarization_error_handling(self, mock_client, sample_page_image):
        """Test error handling during summarization."""
        mock_client.responses.create.side_effect = Exception("API error")

        summary, tokens = summarize_page_image(
            sample_page_image,
            page_number=5,
            document_name="Test Doc"
        )

        assert "Page 5" in summary
        assert "API error" in summary
        assert tokens == 0


class TestSummarizeDocumentFromPages:
    """Tests for summarize_document_from_pages function."""

    @patch('legal_preprocessing.client')
    def test_successful_document_summary(self, mock_client):
        """Test successful document-level summary."""
        mock_response = MagicMock()
        mock_response.output_text = "This is a comprehensive contract covering..."
        mock_client.responses.create.return_value = mock_response

        page_summaries = [
            (1, "Cover page"),
            (2, "Definitions section"),
            (3, "Terms and conditions")
        ]

        summary, tokens = summarize_document_from_pages(
            page_summaries,
            document_name="Test Contract",
            document_type="Contract"
        )

        assert summary == "This is a comprehensive contract covering..."
        assert tokens > 0
        mock_client.responses.create.assert_called_once()

    @patch('legal_preprocessing.client')
    def test_document_summary_prompt_content(self, mock_client):
        """Test that the prompt contains page summaries."""
        mock_response = MagicMock()
        mock_response.output_text = "Summary"
        mock_client.responses.create.return_value = mock_response

        page_summaries = [
            (1, "Page one content"),
            (2, "Page two content")
        ]

        summarize_document_from_pages(
            page_summaries,
            document_name="My Document",
            document_type="Agreement"
        )

        call_args = mock_client.responses.create.call_args
        prompt = call_args.kwargs['input']
        assert "My Document" in prompt
        assert "Agreement" in prompt
        assert "Page 1: Page one content" in prompt

    @patch('legal_preprocessing.client')
    def test_document_summary_error_handling(self, mock_client):
        """Test error handling during document summary."""
        mock_client.responses.create.side_effect = Exception("API failure")

        page_summaries = [(1, "Summary")]

        summary, tokens = summarize_document_from_pages(
            page_summaries,
            document_name="Error Doc",
            document_type="Contract"
        )

        assert "Error Doc" in summary
        assert "1 pages" in summary
        assert tokens == 0


class TestProcessDataRoom:
    """Tests for process_data_room function."""

    @patch('legal_preprocessing.client')
    @patch('legal_preprocessing.convert_from_path')
    def test_successful_data_room_processing(self, mock_convert, mock_client, tmp_path):
        """Test successful processing of a data room."""
        # Set up mock PDF images
        from PIL import Image
        mock_images = [Image.new('RGB', (100, 100), 'white')]
        mock_convert.return_value = mock_images

        # Set up mock API responses
        mock_response = MagicMock()
        mock_response.output_text = "Test summary"
        mock_client.responses.create.return_value = mock_response

        # Create test PDF directory with a PDF file
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        pdf_file = pdf_dir / "test_doc.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        output_dir = tmp_path / "output"

        result = process_data_room(
            str(pdf_dir),
            str(output_dir),
            rate_limit_delay=0
        )

        assert isinstance(result, DataRoom)
        assert result.total_documents == 1
        assert result.total_pages == 1
        assert (output_dir / "data_room_index.json").exists()

    @patch('legal_preprocessing.client')
    @patch('legal_preprocessing.convert_from_path')
    def test_data_room_with_type_mapping(self, mock_convert, mock_client, tmp_path):
        """Test data room processing with document type mapping."""
        from PIL import Image
        mock_convert.return_value = [Image.new('RGB', (100, 100), 'white')]

        mock_response = MagicMock()
        mock_response.output_text = "Summary"
        mock_client.responses.create.return_value = mock_response

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "Articles.pdf").write_bytes(b"content")

        output_dir = tmp_path / "output"

        type_mapping = {"Articles": "Corporate Governance"}

        result = process_data_room(
            str(pdf_dir),
            str(output_dir),
            document_type_mapping=type_mapping,
            rate_limit_delay=0
        )

        assert result.documents[0]["document_type"] == "Corporate Governance"

    @patch('legal_preprocessing.client')
    @patch('legal_preprocessing.convert_from_path')
    def test_empty_pdf_directory(self, mock_convert, mock_client, tmp_path):
        """Test handling of empty PDF directory."""
        pdf_dir = tmp_path / "empty_pdfs"
        pdf_dir.mkdir()
        output_dir = tmp_path / "output"

        result = process_data_room(
            str(pdf_dir),
            str(output_dir),
            rate_limit_delay=0
        )

        assert result.total_documents == 0
        assert result.total_pages == 0

    @patch('legal_preprocessing.client')
    @patch('legal_preprocessing.convert_from_path')
    def test_pdf_extraction_failure_skips_document(self, mock_convert, mock_client, tmp_path):
        """Test that PDF extraction failure skips the document."""
        mock_convert.return_value = []  # Simulate extraction failure

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "bad.pdf").write_bytes(b"content")

        output_dir = tmp_path / "output"

        result = process_data_room(
            str(pdf_dir),
            str(output_dir),
            rate_limit_delay=0
        )

        assert result.total_documents == 0

    @patch('legal_preprocessing.client')
    @patch('legal_preprocessing.convert_from_path')
    def test_multiple_pdfs_processing(self, mock_convert, mock_client, tmp_path):
        """Test processing multiple PDF files."""
        from PIL import Image
        mock_convert.return_value = [Image.new('RGB', (100, 100), 'white')]

        mock_response = MagicMock()
        mock_response.output_text = "Summary"
        mock_client.responses.create.return_value = mock_response

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "doc1.pdf").write_bytes(b"content1")
        (pdf_dir / "doc2.pdf").write_bytes(b"content2")
        (pdf_dir / "doc3.pdf").write_bytes(b"content3")

        output_dir = tmp_path / "output"

        result = process_data_room(
            str(pdf_dir),
            str(output_dir),
            rate_limit_delay=0
        )

        assert result.total_documents == 3

    @patch('legal_preprocessing.client')
    @patch('legal_preprocessing.convert_from_path')
    def test_page_images_saved_correctly(self, mock_convert, mock_client, tmp_path):
        """Test that page images are saved to correct locations."""
        from PIL import Image
        mock_convert.return_value = [
            Image.new('RGB', (100, 100), 'white'),
            Image.new('RGB', (100, 100), 'gray')
        ]

        mock_response = MagicMock()
        mock_response.output_text = "Summary"
        mock_client.responses.create.return_value = mock_response

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "test.pdf").write_bytes(b"content")

        output_dir = tmp_path / "output"

        process_data_room(
            str(pdf_dir),
            str(output_dir),
            rate_limit_delay=0
        )

        # Check images were saved
        images_dir = output_dir / "page_images" / "doc_001"
        assert images_dir.exists()
        assert (images_dir / "page_001.png").exists()
        assert (images_dir / "page_002.png").exists()


class TestDataRoomIntegration:
    """Integration tests for the complete preprocessing pipeline."""

    @patch('legal_preprocessing.client')
    @patch('legal_preprocessing.convert_from_path')
    def test_full_pipeline_integration(self, mock_convert, mock_client, tmp_path):
        """Test the full preprocessing pipeline from PDFs to JSON index."""
        from PIL import Image

        # Set up mocks
        mock_convert.return_value = [
            Image.new('RGB', (100, 100), 'white'),
            Image.new('RGB', (100, 100), 'gray')
        ]

        page_summary_response = MagicMock()
        page_summary_response.output_text = "Page content summary"

        doc_summary_response = MagicMock()
        doc_summary_response.output_text = "Document overview summary"

        # Alternate between page and document summaries
        mock_client.responses.create.side_effect = [
            page_summary_response,
            page_summary_response,
            doc_summary_response
        ]

        # Create PDF directory
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "Test_Document.pdf").write_bytes(b"pdf content")

        output_dir = tmp_path / "output"

        # Process data room
        result = process_data_room(
            str(pdf_dir),
            str(output_dir),
            document_type_mapping={"Test_Document": "Contract"},
            rate_limit_delay=0
        )

        # Verify results
        assert result.total_documents == 1
        assert result.total_pages == 2
        assert result.documents[0]["document_type"] == "Contract"
        assert result.documents[0]["title"] == "Test_Document"

        # Verify JSON index
        with open(output_dir / "data_room_index.json", 'r') as f:
            index_data = json.load(f)

        assert index_data["metadata"]["total_documents"] == 1
        assert index_data["metadata"]["total_pages"] == 2
        assert len(index_data["documents"]) == 1
        assert len(index_data["documents"][0]["pages"]) == 2
