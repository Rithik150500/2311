# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Legal Risk Analysis System built with LangGraph and OpenAI. It preprocesses legal document data rooms (PDFs) into structured summaries, then uses a hierarchical multi-agent architecture to perform due diligence analysis and generate professional reports.

## Architecture

### Core Components

1. **Preprocessing Pipeline** (`legal_preprocessing.py`)
   - Converts PDFs to page images using pdf2image
   - Uses OpenAI vision models (gpt-4.1-mini) to generate page and document summaries
   - Outputs JSON index with multi-tiered summaries for efficient agent navigation

2. **Storage Layer** (`storage_and_tools.py`)
   - `DataRoomStorage` class loads preprocessed JSON index
   - Provides three access levels: document summaries, page summaries, page images
   - LangChain tools wrap storage methods for agent use
   - Includes web research tools (web_search, web_fetch) with usage limits

3. **Agent System** (`agent_configuration.py`)
   - Uses `deepagents` framework with `create_deep_agent()`
   - Main coordinating agent delegates to specialized subagents
   - Two subagents: `legal-analysis-specialist` (detailed analysis) and `report-formatter` (final report)
   - Human-in-the-loop via `interrupt_on` configuration for tool approvals

4. **Approval Workflow** (`approval_workflow.py`)
   - `ApprovalHandler` class manages human-in-the-loop approvals
   - Uses LangGraph's `Command` for resuming execution with decisions
   - Supports approve, reject, and edit decisions

5. **Main Application** (`main_application.py`)
   - Entry point: `run_legal_risk_analysis(data_room_path, output_directory)`
   - Manages the approval loop with iteration safety limit (50)

### Agent Hierarchy

- **Main Agent**: Strategic coordination only, no direct document access
- **Analysis Subagent**: Performs detailed analysis with document/web tools, has usage limits (50 pages, 20 web fetches)
- **Report Subagent**: Formats findings into Word document, invoked once at end

## Running the Application

### Prerequisites
- `OPENAI_API_KEY` environment variable
- `SEARCH_API_KEY` environment variable (for Tavily search API)

### Preprocessing
```python
python legal_preprocessing.py
```
Expects PDFs in `./data_room_pdfs/`, outputs to `./preprocessed_data_room/`

### Analysis
```python
python main_application.py
```
Uses preprocessed data from `./preprocessed_data_room/`, outputs to `./analysis_results/`

## Key Dependencies

- `openai` - Vision model API for document summarization
- `langgraph` - Agent orchestration and checkpointing
- `langchain` - Tool definitions
- `deepagents` - Multi-agent framework
- `pdf2image` - PDF to image conversion
- `Pillow` - Image processing
- `requests`, `beautifulsoup4` - Web fetching

## Important Patterns

### Tool Usage Limits
- Page retrievals: 50 per session (`_PAGE_RETRIEVAL_LIMIT`)
- Web fetches: 20 per session (`_WEB_FETCH_LIMIT`)

### Human-in-the-Loop
Tools requiring approval are configured in `interrupt_on` dictionaries:
- `get_documents`, `web_fetch`, `write_file`, `edit_file` for analysis subagent
- `write_file` for report subagent
- `write_todos`, `task` for main agent

### Storage Initialization
`DataRoomStorage` is initialized as a module-level singleton with hardcoded paths. Modify paths in `storage_and_tools.py` lines 149-152 if needed.
