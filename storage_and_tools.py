"""
storage_and_tools.py

Implements the storage layer and data room access tools that agents use
to navigate the preprocessed document corpus.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import base64


class DataRoomStorage:
    """
    Manages access to the preprocessed data room structure.
    
    This class loads the JSON index created by preprocessing and provides
    efficient query methods that tools can use. It serves as the bridge
    between the preprocessed data and the agent tools.
    """
    
    def __init__(self, index_path: str, base_directory: str):
        """
        Initialize storage by loading the data room index.
        
        Args:
            index_path: Path to data_room_index.json
            base_directory: Base directory containing page images
        """
        self.base_directory = Path(base_directory)
        
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.metadata = data.get('metadata', {})
        self.documents = {
            doc['document_id']: doc 
            for doc in data['documents']
        }
        self.document_list = data['documents']
        
        print(f"Loaded data room with {len(self.documents)} documents, "
              f"{self.metadata.get('total_pages', 0)} pages")
    
    def list_all_documents(self) -> List[Dict]:
        """
        Return all documents with their IDs and summary descriptions.
        
        This provides the high-level overview that agents use to understand
        what exists in the data room before deciding what to examine.
        """
        return [
            {
                'id': doc['document_id'],
                'title': doc['title'],
                'summary_description': doc['summary_description'],
                'document_type': doc['document_type'],
                'page_count': doc['page_count']
            }
            for doc in self.document_list
        ]
    
    def get_document_page_summaries(self, document_ids: List[str]) -> Dict:
        """
        Return page-by-page summaries for specified documents.
        
        This is the medium-fidelity access that lets agents see what each
        page contains without retrieving full images.
        """
        result = {}
        for doc_id in document_ids:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                result[doc_id] = {
                    'title': doc['title'],
                    'document_type': doc['document_type'],
                    'summary': doc['summary_description'],
                    'pages': [
                        {
                            'page_number': p['page_number'],
                            'summary': p['summary_description']
                        }
                        for p in doc['pages']
                    ]
                }
        return result
    
    def get_page_images(
        self,
        document_id: str,
        page_numbers: List[int]
    ) -> Dict:
        """
        Return base64-encoded images for specified pages.
        
        This is the highest-fidelity access, returning actual page images
        that vision models can examine in detail.
        """
        if document_id not in self.documents:
            return {'error': f'Document {document_id} not found'}
        
        doc = self.documents[document_id]
        pages_dict = {p['page_number']: p for p in doc['pages']}
        
        result = {'document_id': document_id, 'title': doc['title'], 'pages': []}
        
        for page_num in page_numbers:
            if page_num not in pages_dict:
                result['pages'].append({
                    'page_number': page_num,
                    'error': f'Page {page_num} not found in document'
                })
                continue
            
            page_info = pages_dict[page_num]
            image_rel_path = page_info['image_path']
            image_full_path = self.base_directory / image_rel_path
            
            if not image_full_path.exists():
                result['pages'].append({
                    'page_number': page_num,
                    'error': f'Image file not found: {image_full_path}'
                })
                continue
            
            # Load and encode image
            try:
                with open(image_full_path, 'rb') as f:
                    image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                result['pages'].append({
                    'page_number': page_num,
                    'summary': page_info['summary_description'],
                    'image_data': f"data:image/png;base64,{image_base64}"
                })
            except Exception as e:
                result['pages'].append({
                    'page_number': page_num,
                    'error': f'Error loading image: {str(e)}'
                })
        
        return result


# Initialize storage (do this once when application starts)
data_room_storage = DataRoomStorage(
    index_path="./preprocessed_data_room/data_room_index.json",
    base_directory="./preprocessed_data_room"
)



























"""
Data room access tools for legal analysis agents.

These tools provide controlled access to preprocessed documents through
three levels of detail: document summaries, page summaries, and page images.
"""

from langchain.tools import tool


@tool
def list_data_room_documents() -> str:
    """
    List all documents in the data room with their IDs and summary descriptions.
    
    Use this tool first when beginning analysis to understand what documents
    are available. The document summaries give you a high-level overview of
    each document's purpose and content.
    
    This tool has no usage limits and does not require approval.
    """
    documents = data_room_storage.list_all_documents()
    
    if not documents:
        return "No documents found in data room."
    
    output_lines = [
        "Available Documents in Data Room:",
        "=" * 60,
        ""
    ]
    
    for doc in documents:
        output_lines.extend([
            f"Document ID: {doc['id']}",
            f"Title: {doc['title']}",
            f"Type: {doc['document_type']}",
            f"Pages: {doc['page_count']}",
            f"Summary: {doc['summary_description']}",
            ""
        ])
    
    return "\n".join(output_lines)


@tool
def get_documents(document_ids: List[str]) -> str:
    """
    Retrieve page-by-page summaries for specified documents.
    
    After reviewing document summaries from list_data_room_documents, use this
    tool to get detailed page-level information for documents that appear relevant
    to your analysis. Each page summary describes what that page contains.
    
    This tool requires human approval before execution. You will see the list of
    documents you requested, and you can add or remove documents before proceeding.
    
    Args:
        document_ids: List of document IDs to retrieve (e.g., ["doc_001", "doc_005"])
    
    Returns:
        Page-by-page summaries for the requested documents
    """
    if not document_ids:
        return "Error: Please provide at least one document ID"
    
    result = data_room_storage.get_document_page_summaries(document_ids)
    
    if not result:
        return f"Error: None of the requested documents were found: {document_ids}"
    
    output_lines = ["Retrieved Document Details:", "=" * 60, ""]
    
    for doc_id, doc_data in result.items():
        output_lines.extend([
            f"Document: {doc_data['title']} ({doc_id})",
            f"Type: {doc_data['document_type']}",
            f"Summary: {doc_data['summary']}",
            "",
            "Page-by-Page Summaries:",
            "-" * 40
        ])
        
        for page in doc_data['pages']:
            output_lines.extend([
                f"  Page {page['page_number']}:",
                f"    {page['summary']}",
                ""
            ])
        
        output_lines.append("=" * 60)
        output_lines.append("")
    
    return "\n".join(output_lines)


# Track page retrieval usage
_page_retrieval_count = 0
_PAGE_RETRIEVAL_LIMIT = 50


@tool
def get_document_pages(document_id: str, page_numbers: List[int]) -> str:
    """
    Retrieve actual page images for detailed examination.
    
    After reviewing page summaries, use this tool to get the full page images
    for specific pages that require detailed legal analysis. The images are
    returned in a format that vision models can examine.
    
    IMPORTANT: This tool has a usage limit of 50 total page retrievals per
    analysis session. Use this tool strategically for only the most critical
    pages that truly need detailed examination.
    
    Args:
        document_id: The document ID (e.g., "doc_001")
        page_numbers: List of specific page numbers to retrieve (e.g., [1, 5, 12])
    
    Returns:
        Page images with their summaries for context
    """
    global _page_retrieval_count
    
    if not page_numbers:
        return "Error: Please specify at least one page number"
    
    # Check usage limit
    if _page_retrieval_count >= _PAGE_RETRIEVAL_LIMIT:
        return (f"Error: Page retrieval limit reached ({_PAGE_RETRIEVAL_LIMIT} pages). "
                f"You have already retrieved the maximum number of pages allowed. "
                f"Consider which previously retrieved pages contain the information you need.")
    
    remaining_quota = _PAGE_RETRIEVAL_LIMIT - _page_retrieval_count
    
    if len(page_numbers) > remaining_quota:
        return (f"Error: Requesting {len(page_numbers)} pages would exceed the limit. "
                f"You have {remaining_quota} page retrievals remaining. "
                f"Please reduce the number of pages requested.")
    
    # Retrieve pages
    result = data_room_storage.get_page_images(document_id, page_numbers)
    
    if 'error' in result:
        return f"Error: {result['error']}"
    
    # Update usage counter
    successful_retrievals = len([p for p in result['pages'] if 'image_data' in p])
    _page_retrieval_count += successful_retrievals
    
    # Format output
    output_lines = [
        f"Retrieved {successful_retrievals} pages from {result['title']} ({document_id})",
        f"Remaining page retrieval quota: {_PAGE_RETRIEVAL_LIMIT - _page_retrieval_count}",
        "=" * 60,
        ""
    ]
    
    for page_data in result['pages']:
        if 'error' in page_data:
            output_lines.extend([
                f"Page {page_data['page_number']}: ERROR - {page_data['error']}",
                ""
            ])
        else:
            output_lines.extend([
                f"Page {page_data['page_number']}:",
                f"Summary: {page_data['summary']}",
                f"Image: [Base64 image data available for vision analysis]",
                ""
            ])
    
    return "\n".join(output_lines)






























"""
web_research_tools.py

Implements web search and fetch capabilities that allow agents to research
external legal context, statutes, regulations, and case law.
"""

from langchain.tools import tool
import requests
from typing import Optional
import os

# You would typically use a proper search API like Tavily, Brave, or similar
# This is a simplified example showing the pattern
SEARCH_API_KEY = os.environ.get("SEARCH_API_KEY")
SEARCH_API_URL = "https://api.tavily.com/search"

# Track web fetch usage
_web_fetch_count = 0
_WEB_FETCH_LIMIT = 20


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for legal information, statutes, regulations, or case law.
    
    Use this tool to find external legal context that helps interpret or
    analyze provisions found in the data room documents. Search for:
    - Applicable statutes and regulations
    - Legal standards and requirements
    - Case law and legal precedents
    - Industry best practices
    - Regulatory guidance
    
    This tool returns search results with titles, URLs, and snippets. Review
    the results to identify authoritative sources, then use web_fetch to
    retrieve full content from the most relevant sources.
    
    This tool has no usage limits and does not require approval.
    
    Args:
        query: Search query (e.g., "Delaware corporate governance requirements")
        max_results: Maximum number of results to return (default 5)
    
    Returns:
        Formatted search results with titles, URLs, and snippets
    """
    if not query.strip():
        return "Error: Please provide a search query"
    
    try:
        # Make the search API call
        # This is a simplified example - in production you would use a real API
        response = requests.post(
            SEARCH_API_URL,
            json={
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced"
            },
            headers={"Authorization": f"Bearer {SEARCH_API_KEY}"},
            timeout=10
        )
        
        if response.status_code != 200:
            return f"Error: Search request failed with status {response.status_code}"
        
        results = response.json().get("results", [])
        
        if not results:
            return f"No results found for query: {query}"
        
        # Format results for the agent
        output_lines = [
            f"Search Results for: {query}",
            "=" * 60,
            ""
        ]
        
        for idx, result in enumerate(results, start=1):
            output_lines.extend([
                f"{idx}. {result.get('title', 'Untitled')}",
                f"   URL: {result.get('url', 'N/A')}",
                f"   Snippet: {result.get('snippet', 'No snippet available')}",
                f"   Source: {result.get('domain', 'Unknown')}",
                ""
            ])
        
        output_lines.append(
            "Use web_fetch with specific URLs to retrieve full content from authoritative sources."
        )
        
        return "\n".join(output_lines)
        
    except requests.Timeout:
        return "Error: Search request timed out. Please try again."
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def web_fetch(url: str) -> str:
    """
    Fetch the complete content of a specific web page.
    
    After using web_search to identify relevant sources, use this tool to
    retrieve the full content of authoritative pages. Focus on:
    - Government and regulatory websites
    - Official legal databases
    - Court websites and legal repositories
    - Reputable legal analysis and commentary
    
    IMPORTANT: This tool has a usage limit of 20 fetches per analysis session.
    It also requires human approval before execution. Be selective about which
    sources you retrieve - prioritize official and authoritative sources over
    secondary sources or general information sites.
    
    Args:
        url: The complete URL to fetch (e.g., "https://www.sec.gov/rules/...")
    
    Returns:
        The text content of the web page, with HTML stripped for readability
    """
    global _web_fetch_count
    
    if not url.strip():
        return "Error: Please provide a URL to fetch"
    
    # Check usage limit
    if _web_fetch_count >= _WEB_FETCH_LIMIT:
        return (f"Error: Web fetch limit reached ({_WEB_FETCH_LIMIT} fetches). "
                f"You have already retrieved the maximum number of web pages allowed. "
                f"Review the content you have already fetched.")
    
    remaining = _WEB_FETCH_LIMIT - _web_fetch_count
    
    try:
        # Fetch the web page
        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; LegalAnalysisBot/1.0)"
            }
        )
        
        if response.status_code != 200:
            return f"Error: Failed to fetch URL (status {response.status_code}): {url}"
        
        # Extract text content (you would use a proper HTML parser in production)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Truncate if extremely long
        max_length = 8000
        if len(text) > max_length:
            text = text[:max_length] + f"\n\n[Content truncated at {max_length} characters]"
        
        # Update usage counter
        _web_fetch_count += 1
        
        output = [
            f"Fetched content from: {url}",
            f"Remaining fetch quota: {remaining - 1}",
            "=" * 60,
            "",
            text
        ]
        
        return "\n".join(output)
        
    except requests.Timeout:
        return f"Error: Request timed out while fetching: {url}"
    except Exception as e:
        return f"Error fetching URL: {str(e)}"