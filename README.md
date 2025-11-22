# Legal Risk Analysis System

A hierarchical multi-agent system for automated legal due diligence analysis. This system preprocesses legal document data rooms (PDFs) into structured summaries, then uses specialized AI agents to perform comprehensive risk analysis and generate professional reports.

## Features

- **Intelligent Document Preprocessing**: Converts PDFs to page images and generates multi-tiered summaries using OpenAI vision models
- **Hierarchical Agent Architecture**: Main coordinating agent delegates to specialized subagents for focused analysis
- **Human-in-the-Loop Approvals**: Configurable approval workflows for critical operations
- **Web Research Integration**: Agents can search and fetch external legal context, regulations, and case law
- **Professional Report Generation**: Automated creation of comprehensive legal risk analysis reports in Word format
- **Resource Management**: Built-in usage limits to control costs (50 page retrievals, 20 web fetches per session)

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────┐
│                   Main Agent                         │
│            (Strategic Coordination)                  │
│                                                      │
│  • Plans analysis strategy                           │
│  • Delegates to subagents                           │
│  • Synthesizes findings                             │
└─────────────────┬───────────────┬───────────────────┘
                  │               │
          ┌───────▼───────┐ ┌─────▼───────┐
          │   Analysis    │ │   Report    │
          │  Specialist   │ │  Formatter  │
          │               │ │             │
          │ • Doc access  │ │ • Formats   │
          │ • Web research│ │   findings  │
          │ • Risk ID     │ │ • Creates   │
          └───────────────┘ │   Word doc  │
                            └─────────────┘
```

### Core Modules

| Module | Description |
|--------|-------------|
| `legal_preprocessing.py` | Converts PDFs to images and generates page/document summaries |
| `storage_and_tools.py` | Storage layer and LangChain tools for document access |
| `agent_configuration.py` | Defines agent hierarchy using deepagents framework |
| `approval_workflow.py` | Human-in-the-loop approval handling |
| `main_application.py` | Entry point that orchestrates the complete workflow |

## Prerequisites

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export SEARCH_API_KEY="your-tavily-api-key"
```

### System Dependencies

- Python 3.8+
- Poppler (for PDF to image conversion)

#### Installing Poppler

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**Windows:**
Download from [poppler releases](https://github.com/osber/poppler-windows/releases)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd legal-risk-analysis
```

2. Install Python dependencies:
```bash
pip install openai langgraph langchain deepagents pdf2image Pillow requests beautifulsoup4
```

3. Set up environment variables (see Prerequisites)

## Usage

### Step 1: Preprocess Documents

Place your PDF documents in `./data_room_pdfs/`, then run:

```bash
python legal_preprocessing.py
```

This will:
- Extract pages from each PDF as images
- Generate page-level summaries using OpenAI vision
- Create document-level summaries
- Output structured data to `./preprocessed_data_room/`

**Optional**: Configure document type mapping in the script:

```python
document_types = {
    "Articles_of_Incorporation": "Corporate Governance",
    "Master_Services_Agreement": "Contracts",
    "Patent_Portfolio": "Intellectual Property",
    # Add your documents here
}
```

### Step 2: Run Analysis

```bash
python main_application.py
```

This will:
- Load the preprocessed data room
- Initialize the agent system
- Run the hierarchical analysis workflow
- Prompt for approvals at configured checkpoints
- Generate a final risk analysis report

### Output

Results are saved to `./analysis_results/` including:
- Individual risk domain findings (text files)
- Final comprehensive report (Word document)

## Project Structure

```
.
├── data_room_pdfs/           # Input: PDF documents to analyze
├── preprocessed_data_room/   # Intermediate: Processed summaries and images
│   ├── data_room_index.json  # Document and page metadata
│   └── page_images/          # Extracted page images
│       └── doc_001/
│           ├── page_001.png
│           └── ...
├── analysis_results/         # Output: Analysis findings and reports
├── legal_preprocessing.py    # PDF preprocessing pipeline
├── storage_and_tools.py      # Storage layer and agent tools
├── agent_configuration.py    # Agent system configuration
├── approval_workflow.py      # Human-in-the-loop handling
├── main_application.py       # Main entry point
├── CLAUDE.md                 # Development guidance
└── README.md                 # This file
```

## Configuration

### Storage Paths

Modify paths in `storage_and_tools.py` (lines 149-152):

```python
data_room_storage = DataRoomStorage(
    index_path="./preprocessed_data_room/data_room_index.json",
    base_directory="./preprocessed_data_room"
)
```

### Tool Approval Configuration

Configure which tools require human approval in `agent_configuration.py`:

```python
# Analysis subagent approvals
"interrupt_on": {
    "get_documents": True,
    "web_fetch": True,
    "write_file": True,
    "edit_file": True
}

# Main agent approvals
"interrupt_on": {
    "write_todos": True,
    "task": True,
}
```

### Usage Limits

Adjust limits in `storage_and_tools.py`:

```python
_PAGE_RETRIEVAL_LIMIT = 50  # Maximum page images per session
_WEB_FETCH_LIMIT = 20       # Maximum web fetches per session
```

## Agent Workflow

1. **Strategic Planning**: Main agent creates analysis plan based on document summaries
2. **Delegation**: Tasks are delegated to analysis specialist for each risk domain
3. **Document Analysis**: Subagent navigates documents using three access levels:
   - Document summaries (no approval needed)
   - Page summaries (requires approval)
   - Page images (limited to 50 per session)
4. **Web Research**: External research for legal context (limited to 20 fetches)
5. **Findings Documentation**: Results written to files with citations
6. **Report Generation**: Report formatter creates final Word document

## Risk Categories Analyzed

- Corporate Governance
- Commercial Contracts
- Intellectual Property
- Regulatory Compliance
- Employment Matters
- Litigation and Disputes
- Financial Arrangements
- Real Estate Holdings
- Environmental Compliance

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `openai` | Vision model API for document summarization |
| `langgraph` | Agent orchestration and checkpointing |
| `langchain` | Tool definitions |
| `deepagents` | Multi-agent framework |
| `pdf2image` | PDF to image conversion |
| `Pillow` | Image processing |
| `requests` | HTTP requests for web research |
| `beautifulsoup4` | HTML parsing for web content |

## Troubleshooting

### Common Issues

**PDF extraction fails:**
- Ensure Poppler is installed and in PATH
- Check PDF file is not corrupted or password-protected

**API rate limits:**
- Adjust `rate_limit_delay` in `process_data_room()`
- Default is 0.1 seconds between calls

**Memory issues with large PDFs:**
- Reduce DPI in `extract_pages_from_pdf()` (default 200)
- Process documents in smaller batches

**Missing environment variables:**
- Verify `OPENAI_API_KEY` and `SEARCH_API_KEY` are set
- Check for typos in variable names

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license here]
