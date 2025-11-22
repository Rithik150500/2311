"""
test_agent_configuration.py

Tests for the agent configuration module that sets up the legal analysis agent system.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCreateLegalAnalysisSystem:
    """Tests for create_legal_analysis_system function."""

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_returns_agent_and_checkpointer(self, mock_saver, mock_create_agent):
        """Test that function returns agent and checkpointer tuple."""
        mock_agent = MagicMock()
        mock_checkpointer = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_saver.return_value = mock_checkpointer

        from agent_configuration import create_legal_analysis_system
        agent, checkpointer = create_legal_analysis_system()

        assert agent == mock_agent
        assert checkpointer == mock_checkpointer

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_creates_memory_saver(self, mock_saver, mock_create_agent):
        """Test that MemorySaver is instantiated."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        mock_saver.assert_called_once()

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_configures_main_agent_model(self, mock_saver, mock_create_agent):
        """Test that main agent uses correct model."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        assert call_kwargs['model'] == 'claude-sonnet-4-5-20250929'

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_main_agent_has_no_direct_tools(self, mock_saver, mock_create_agent):
        """Test that main agent has no direct data room tools."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        assert call_kwargs['tools'] == []

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_configures_two_subagents(self, mock_saver, mock_create_agent):
        """Test that two subagents are configured."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']
        assert len(subagents) == 2

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_analysis_subagent_configuration(self, mock_saver, mock_create_agent):
        """Test analysis subagent has correct configuration."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        # Find analysis subagent
        analysis_subagent = next(
            s for s in subagents if s['name'] == 'legal-analysis-specialist'
        )

        assert analysis_subagent['model'] == 'claude-sonnet-4-5-20250929'
        assert len(analysis_subagent['tools']) == 5
        assert 'interrupt_on' in analysis_subagent

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_analysis_subagent_tools(self, mock_saver, mock_create_agent):
        """Test analysis subagent has all required tools."""
        from agent_configuration import create_legal_analysis_system
        from storage_and_tools import (
            list_data_room_documents,
            get_documents,
            get_document_pages,
        )

        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        analysis_subagent = next(
            s for s in subagents if s['name'] == 'legal-analysis-specialist'
        )

        tools = analysis_subagent['tools']
        assert list_data_room_documents in tools
        assert get_documents in tools
        assert get_document_pages in tools
        # web_search and web_fetch are from mocked web_research_tools
        assert len(tools) == 5

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_analysis_subagent_interrupt_config(self, mock_saver, mock_create_agent):
        """Test analysis subagent interrupt configuration."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        analysis_subagent = next(
            s for s in subagents if s['name'] == 'legal-analysis-specialist'
        )

        interrupt_on = analysis_subagent['interrupt_on']
        assert interrupt_on.get('get_documents') is True
        assert interrupt_on.get('web_fetch') is True
        assert interrupt_on.get('write_file') is True
        assert interrupt_on.get('edit_file') is True

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_report_subagent_configuration(self, mock_saver, mock_create_agent):
        """Test report subagent has correct configuration."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        # Find report subagent
        report_subagent = next(
            s for s in subagents if s['name'] == 'report-formatter'
        )

        assert report_subagent['model'] == 'claude-sonnet-4-5-20250929'
        assert len(report_subagent['tools']) == 1  # Only list_data_room_documents

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_report_subagent_tools(self, mock_saver, mock_create_agent):
        """Test report subagent has limited tools."""
        from agent_configuration import create_legal_analysis_system
        from storage_and_tools import list_data_room_documents

        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        report_subagent = next(
            s for s in subagents if s['name'] == 'report-formatter'
        )

        tools = report_subagent['tools']
        assert list_data_room_documents in tools
        # Should NOT have document retrieval or web research tools

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_report_subagent_interrupt_config(self, mock_saver, mock_create_agent):
        """Test report subagent interrupt configuration."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        report_subagent = next(
            s for s in subagents if s['name'] == 'report-formatter'
        )

        interrupt_on = report_subagent['interrupt_on']
        assert interrupt_on.get('write_file') is True

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_main_agent_interrupt_config(self, mock_saver, mock_create_agent):
        """Test main agent interrupt configuration."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        interrupt_on = call_kwargs['interrupt_on']

        assert interrupt_on.get('write_todos') is True
        assert interrupt_on.get('task') is True

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_checkpointer_passed_to_agent(self, mock_saver, mock_create_agent):
        """Test that checkpointer is passed to agent creation."""
        mock_checkpointer = MagicMock()
        mock_saver.return_value = mock_checkpointer

        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        assert call_kwargs['checkpointer'] == mock_checkpointer


class TestSubagentDescriptions:
    """Tests for subagent description content."""

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_analysis_subagent_description(self, mock_saver, mock_create_agent):
        """Test analysis subagent has appropriate description."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        analysis_subagent = next(
            s for s in subagents if s['name'] == 'legal-analysis-specialist'
        )

        description = analysis_subagent['description']
        assert 'legal risk analysis' in description.lower()
        assert 'documents' in description.lower()

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_report_subagent_description(self, mock_saver, mock_create_agent):
        """Test report subagent has appropriate description."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        report_subagent = next(
            s for s in subagents if s['name'] == 'report-formatter'
        )

        description = report_subagent['description']
        assert 'report' in description.lower()
        assert 'once' in description.lower()  # Should only be invoked once


class TestSystemPrompts:
    """Tests for system prompt content."""

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_main_agent_system_prompt(self, mock_saver, mock_create_agent):
        """Test main agent system prompt contains key instructions."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        system_prompt = call_kwargs['system_prompt']

        assert 'coordinator' in system_prompt.lower()
        assert 'delegate' in system_prompt.lower()
        assert 'strategic' in system_prompt.lower()

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_analysis_subagent_system_prompt(self, mock_saver, mock_create_agent):
        """Test analysis subagent system prompt contains key instructions."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        analysis_subagent = next(
            s for s in subagents if s['name'] == 'legal-analysis-specialist'
        )

        system_prompt = analysis_subagent['system_prompt']
        assert 'legal' in system_prompt.lower()
        assert 'analysis' in system_prompt.lower()
        assert '50' in system_prompt  # Page retrieval limit
        assert '20' in system_prompt  # Web fetch limit

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_report_subagent_system_prompt(self, mock_saver, mock_create_agent):
        """Test report subagent system prompt contains key instructions."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        report_subagent = next(
            s for s in subagents if s['name'] == 'report-formatter'
        )

        system_prompt = report_subagent['system_prompt']
        assert 'report' in system_prompt.lower()
        assert 'format' in system_prompt.lower()
        assert 'executive summary' in system_prompt.lower()


class TestModuleLevelAgentCreation:
    """Tests for module-level agent creation."""

    def test_module_exports_agent_and_checkpointer(self):
        """Test that module exports required objects."""
        from agent_configuration import legal_analysis_agent, checkpointer

        # Verify the module exports the required objects
        assert legal_analysis_agent is not None
        assert checkpointer is not None


class TestAgentConfigurationStructure:
    """Tests for the overall structure of agent configuration."""

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_subagent_names_are_unique(self, mock_saver, mock_create_agent):
        """Test that subagent names are unique."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        names = [s['name'] for s in subagents]
        assert len(names) == len(set(names))  # All names unique

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_all_subagents_have_required_fields(self, mock_saver, mock_create_agent):
        """Test that all subagents have required configuration fields."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        required_fields = ['name', 'description', 'system_prompt', 'tools', 'model', 'interrupt_on']

        for subagent in subagents:
            for field in required_fields:
                assert field in subagent, f"Missing {field} in subagent {subagent.get('name')}"


class TestToolAssignments:
    """Tests for correct tool assignments to agents."""

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_analysis_has_document_access_tools(self, mock_saver, mock_create_agent):
        """Test analysis subagent has document access tools."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        analysis_subagent = next(
            s for s in subagents if s['name'] == 'legal-analysis-specialist'
        )

        tool_names = [t.name for t in analysis_subagent['tools']]
        assert 'list_data_room_documents' in tool_names
        assert 'get_documents' in tool_names
        assert 'get_document_pages' in tool_names

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_analysis_has_web_research_tools(self, mock_saver, mock_create_agent):
        """Test analysis subagent has web research tools."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        analysis_subagent = next(
            s for s in subagents if s['name'] == 'legal-analysis-specialist'
        )

        tool_names = [t.name for t in analysis_subagent['tools']]
        assert 'web_search' in tool_names
        assert 'web_fetch' in tool_names

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_report_has_no_web_tools(self, mock_saver, mock_create_agent):
        """Test report subagent does not have web research tools."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        report_subagent = next(
            s for s in subagents if s['name'] == 'report-formatter'
        )

        tool_names = [t.name for t in report_subagent['tools']]
        assert 'web_search' not in tool_names
        assert 'web_fetch' not in tool_names

    @patch('agent_configuration.create_deep_agent')
    @patch('agent_configuration.MemorySaver')
    def test_report_has_no_document_retrieval_tools(self, mock_saver, mock_create_agent):
        """Test report subagent does not have document retrieval tools."""
        from agent_configuration import create_legal_analysis_system
        create_legal_analysis_system()

        call_kwargs = mock_create_agent.call_args.kwargs
        subagents = call_kwargs['subagents']

        report_subagent = next(
            s for s in subagents if s['name'] == 'report-formatter'
        )

        tool_names = [t.name for t in report_subagent['tools']]
        assert 'get_documents' not in tool_names
        assert 'get_document_pages' not in tool_names
