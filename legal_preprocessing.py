"""
legal_preprocessing.py

Preprocesses a data room of PDF documents into a structured format with
multi-tiered summaries that enable efficient agent navigation.
"""

import os
import json
import base64
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pdf2image import convert_from_path
from io import BytesIO
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@dataclass
class PageSummary:
    """Represents a single page's summary with its image location"""
    page_number: int
    summary_description: str
    image_path: str
    tokens_used: int


@dataclass
class DocumentSummary:
    """Represents a complete document with all its pages"""
    document_id: str
    title: str
    document_type: str
    summary_description: str
    page_count: int
    pages: List[Dict]  # Will contain serialized PageSummary objects
    pdf_path: str
    total_tokens: int


@dataclass
class DataRoom:
    """Represents the complete preprocessed data room"""
    documents: List[Dict]  # Will contain serialized DocumentSummary objects
    total_documents: int
    total_pages: int
    total_tokens: int
    
    def to_json(self, output_path: str):
        """Save the complete data room structure to JSON"""
        data = {
            "metadata": {
                "total_documents": self.total_documents,
                "total_pages": self.total_pages,
                "total_tokens": self.total_tokens
            },
            "documents": self.documents
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Data room index saved to {output_path}")


def extract_pages_from_pdf(pdf_path: str, dpi: int = 200) -> List[bytes]:
    """
    Extract all pages from a PDF as individual PNG images.
    
    The resolution of 200 DPI provides good text readability for legal documents
    while keeping file sizes manageable. Each page becomes a separate image that
    can be analyzed by vision models.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for image extraction (200 is good for legal docs)
    
    Returns:
        List of PNG image data as bytes
    """
    print(f"  Extracting pages at {dpi} DPI...")
    
    try:
        # Convert PDF pages to PIL Image objects
        images = convert_from_path(pdf_path, dpi=dpi, fmt='png')
        
        # Convert PIL images to bytes
        page_images = []
        for image in images:
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            page_images.append(buffer.getvalue())
        
        return page_images
        
    except Exception as e:
        print(f"  ✗ Error extracting pages: {e}")
        return []


def calculate_image_tokens(width: int, height: int, model: str = "gpt-4.1-mini") -> int:
    """
    Calculate token cost for an image based on OpenAI's pricing formula.
    
    This implements the exact formula from OpenAI's documentation for GPT-4.1 models:
    1. Calculate 32x32 patches needed to cover the image
    2. If patches exceed 1536, scale down the image
    3. Apply model-specific multiplier
    
    Understanding token costs helps you budget for preprocessing operations.
    """
    multipliers = {
        "gpt-5-mini": 1.62,
        "gpt-5-nano": 2.46,
        "gpt-4.1-mini": 1.62,
        "gpt-4.1-nano": 2.46,
    }
    
    multiplier = multipliers.get(model, 1.62)
    
    # Calculate raw patches
    raw_patches = math.ceil(width / 32) * math.ceil(height / 32)
    
    # If exceeds cap, scale down
    if raw_patches > 1536:
        r = math.sqrt((32 * 32 * 1536) / (width * height))
        resized_width = width * r
        resized_height = height * r
        patches_width = math.floor(resized_width / 32)
        patches_height = math.floor(resized_height / 32)
        image_tokens = patches_width * patches_height
    else:
        image_tokens = raw_patches
    
    # Apply multiplier and return
    return math.ceil(image_tokens * multiplier)


def summarize_page_image(
    page_image_bytes: bytes,
    page_number: int,
    document_name: str,
    model: str = "gpt-4.1-mini"
) -> Tuple[str, int]:
    """
    Use OpenAI vision model to create a concise summary of a page.
    
    This is where the magic happens - a vision-capable language model examines
    the page image and produces a natural language description of what it contains.
    The summary captures the type of content, key topics, notable elements, and
    the page's role in the document without reproducing verbatim text.
    
    Returns:
        Tuple of (summary_text, tokens_used)
    """
    # Convert image bytes to base64 for API
    base64_image = base64.b64encode(page_image_bytes).decode('utf-8')
    
    # Craft a detailed prompt that guides the model toward useful summaries
    prompt = f"""You are analyzing page {page_number} of a legal document titled "{document_name}".

Provide a concise summary description of this page that captures:
1. The type of content (e.g., signature page, terms and conditions, financial data, organizational chart, definitions)
2. Key topics or subjects covered
3. Notable elements like dates, parties named, monetary amounts, or legal provisions
4. The page's role in the document structure (e.g., cover page, main body, exhibits, schedules)

Your summary should be two to three sentences that give a legal analyst enough information to decide whether this page needs detailed examination. 

DO NOT reproduce the actual text verbatim. Instead, describe what the page contains using your own words.

Good example: "Signature page with execution date of March 15, 2023, signed by CEO John Smith and witnessed by corporate secretary. Contains standard representations about authority to execute and no conflicts with other agreements."

Poor example: "This page has text and signatures."

Be specific about what makes this page legally significant or relevant for due diligence."""

    try:
        # Call OpenAI's Responses API with vision
        response = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"  # High detail for legal documents with small text
                    }
                ]
            }],
            reasoning={"effort": "none"},  # No deep reasoning needed for description
            text={"verbosity": "low"}  # Keep summaries concise
        )
        
        # Extract the summary text
        summary = response.output_text
        
        # Calculate tokens used (approximate based on image dimensions)
        # In production, you'd extract this from response.usage
        from PIL import Image
        img = Image.open(BytesIO(page_image_bytes))
        tokens_used = calculate_image_tokens(img.width, img.height, model)
        
        return summary, tokens_used
        
    except Exception as e:
        print(f"  ✗ Error summarizing page {page_number}: {e}")
        return f"Page {page_number} (summary unavailable: {str(e)})", 0


def summarize_document_from_pages(
    page_summaries: List[Tuple[int, str]],
    document_name: str,
    document_type: str,
    model: str = "gpt-4.1-mini"
) -> Tuple[str, int]:
    """
    Create a document-level summary by synthesizing all page summaries.
    
    This step takes dozens of page-level summaries and distills them into a
    single concise characterization of what the document is, why it matters,
    and what key information it contains. This document summary is what agents
    see first when deciding which documents to examine in detail.
    
    Returns:
        Tuple of (summary_text, tokens_used)
    """
    # Format all page summaries into a readable structure
    combined_summaries = "\n\n".join([
        f"Page {page_num}: {summary}"
        for page_num, summary in page_summaries
    ])
    
    prompt = f"""You are analyzing a legal document titled "{document_name}" categorized as "{document_type}".

Below are summaries of each page in the document:

{combined_summaries}

Based on these page summaries, provide a concise document-level summary that captures:
1. The overall purpose and nature of this document
2. Key parties involved (companies, individuals, entities)
3. Main topics, provisions, or sections covered
4. Notable dates, monetary amounts, or other critical details
5. Any unique or unusual aspects that stand out

Your summary should be three to four sentences that give a legal analyst a clear understanding of what this document is and why it might be important for due diligence. This summary will help analysts decide which documents to examine in detail.

Good example: "Master Services Agreement between TechCorp Inc. and Enterprise Client dated January 15, 2023 with three-year term and automatic renewal. Covers software licensing, support obligations, liability caps of five million dollars, and standard indemnification provisions. Includes unusual termination for convenience clause allowing either party to exit with ninety days notice. Annual contract value approximately two point four million dollars."

Be specific and highlight what makes this document legally significant."""

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "none"},
            text={"verbosity": "low"}
        )
        
        summary = response.output_text
        
        # Approximate tokens (in production, use response.usage)
        tokens_used = len(combined_summaries.split()) * 1.3  # Rough estimate
        
        return summary, int(tokens_used)
        
    except Exception as e:
        print(f"  ✗ Error creating document summary: {e}")
        return f"Document {document_name} containing {len(page_summaries)} pages", 0


def process_data_room(
    pdf_directory: str,
    output_directory: str,
    document_type_mapping: Optional[Dict[str, str]] = None,
    model: str = "gpt-4.1-mini",
    rate_limit_delay: float = 0.1
) -> DataRoom:
    """
    Process all PDFs in a directory to create a structured data room.
    
    This is the main orchestration function that coordinates the entire
    preprocessing pipeline. It processes each document sequentially,
    extracting pages, generating summaries, and building the complete
    data room index.
    
    Args:
        pdf_directory: Directory containing PDF files
        output_directory: Where to save processed data and index
        document_type_mapping: Maps document names to categories
        model: Which vision model to use
        rate_limit_delay: Seconds to wait between API calls
    
    Returns:
        Complete DataRoom object with all processed documents
    """
    pdf_dir = Path(pdf_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectory for page images
    images_dir = output_dir / "page_images"
    images_dir.mkdir(exist_ok=True)
    
    # Initialize tracking variables
    documents = []
    total_pages = 0
    total_tokens = 0
    
    # Default document type mapping
    if document_type_mapping is None:
        document_type_mapping = {}
    
    # Process each PDF file
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDF documents to process\n")
    
    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{idx}/{len(pdf_files)}] Processing {pdf_path.name}")
        
        # Generate document ID and metadata
        doc_id = f"doc_{idx:03d}"
        doc_name = pdf_path.stem
        doc_type = document_type_mapping.get(doc_name, "General")
        
        # Create directory for this document's page images
        doc_images_dir = images_dir / doc_id
        doc_images_dir.mkdir(exist_ok=True)
        
        # Extract pages as images
        page_images = extract_pages_from_pdf(str(pdf_path))
        
        if not page_images:
            print(f"  ✗ Skipping document due to extraction failure\n")
            continue
        
        print(f"  Extracted {len(page_images)} pages")
        
        # Process each page
        page_summaries_objects = []
        doc_tokens = 0
        
        for page_num, page_image_bytes in enumerate(page_images, start=1):
            print(f"  Analyzing page {page_num}/{len(page_images)}...", end=" ")
            
            # Save the page image
            image_filename = f"page_{page_num:03d}.png"
            image_path = doc_images_dir / image_filename
            with open(image_path, 'wb') as f:
                f.write(page_image_bytes)
            
            # Get page summary from vision model
            summary, tokens = summarize_page_image(
                page_image_bytes,
                page_num,
                doc_name,
                model
            )
            
            print(f"✓ ({tokens} tokens)")
            
            page_summaries_objects.append(PageSummary(
                page_number=page_num,
                summary_description=summary,
                image_path=str(image_path.relative_to(output_dir)),
                tokens_used=tokens
            ))
            
            doc_tokens += tokens
            
            # Rate limiting to avoid API limits
            time.sleep(rate_limit_delay)
        
        # Create document-level summary
        print(f"  Creating document summary...")
        doc_summary, summary_tokens = summarize_document_from_pages(
            [(p.page_number, p.summary_description) for p in page_summaries_objects],
            doc_name,
            doc_type,
            model
        )
        doc_tokens += summary_tokens
        
        # Create document object
        document = DocumentSummary(
            document_id=doc_id,
            title=doc_name,
            document_type=doc_type,
            summary_description=doc_summary,
            page_count=len(page_summaries_objects),
            pages=[asdict(p) for p in page_summaries_objects],
            pdf_path=str(pdf_path),
            total_tokens=doc_tokens
        )
        
        documents.append(asdict(document))
        total_pages += len(page_summaries_objects)
        total_tokens += doc_tokens
        
        print(f"  ✓ Completed {doc_name} ({doc_tokens} total tokens)\n")
    
    # Create complete data room
    data_room = DataRoom(
        documents=documents,
        total_documents=len(documents),
        total_pages=total_pages,
        total_tokens=total_tokens
    )
    
    # Save to JSON
    index_path = output_dir / "data_room_index.json"
    data_room.to_json(str(index_path))
    
    print(f"\n{'='*60}")
    print(f"Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"Documents processed: {len(documents)}")
    print(f"Total pages analyzed: {total_pages}")
    print(f"Total tokens used: {total_tokens:,}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    return data_room


# Example usage
if __name__ == "__main__":
    # Define document categories for your data room
    document_types = {
        "Articles_of_Incorporation": "Corporate Governance",
        "Bylaws": "Corporate Governance",
        "Master_Services_Agreement": "Contracts",
        "Employment_Agreement_CEO": "Employment",
        "Patent_Portfolio": "Intellectual Property",
        "Financial_Statements_2023": "Financial",
        "Regulatory_Compliance_Report": "Regulatory"
    }
    
    # Process the data room
    data_room = process_data_room(
        pdf_directory="./data_room_pdfs",
        output_directory="./preprocessed_data_room",
        document_type_mapping=document_types,
        model="gpt-4.1-mini"
    )