"""
test_approval_workflow.py

Tests for the approval workflow module that handles human-in-the-loop approvals.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from langgraph.types import Command

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from approval_workflow import ApprovalHandler, approval_handler


class TestApprovalHandlerCheckForInterrupt:
    """Tests for check_for_interrupt method."""

    def test_check_for_interrupt_with_interrupt(self, sample_interrupt_result):
        """Test detecting an interrupt in result."""
        handler = ApprovalHandler()
        assert handler.check_for_interrupt(sample_interrupt_result) is True

    def test_check_for_interrupt_without_interrupt(self):
        """Test when no interrupt is present."""
        handler = ApprovalHandler()
        result = {"messages": [{"content": "Hello"}]}
        assert handler.check_for_interrupt(result) is False

    def test_check_for_interrupt_empty_result(self):
        """Test with empty result."""
        handler = ApprovalHandler()
        assert handler.check_for_interrupt({}) is False

    def test_check_for_interrupt_none_interrupt(self):
        """Test when __interrupt__ is None."""
        handler = ApprovalHandler()
        result = {"__interrupt__": None}
        assert handler.check_for_interrupt(result) is False

    def test_check_for_interrupt_empty_list(self):
        """Test when __interrupt__ is empty list."""
        handler = ApprovalHandler()
        result = {"__interrupt__": []}
        assert handler.check_for_interrupt(result) is False


class TestApprovalHandlerExtractPendingActions:
    """Tests for extract_pending_actions method."""

    def test_extract_pending_actions_success(self, sample_interrupt_result):
        """Test extracting pending actions from interrupt."""
        handler = ApprovalHandler()
        actions = handler.extract_pending_actions(sample_interrupt_result)

        assert len(actions) == 1
        assert actions[0]["name"] == "get_documents"
        assert actions[0]["args"]["document_ids"] == ["doc_001", "doc_002"]

    def test_extract_pending_actions_no_interrupt(self):
        """Test extracting actions when no interrupt."""
        handler = ApprovalHandler()
        result = {"messages": []}
        actions = handler.extract_pending_actions(result)

        assert actions == []

    def test_extract_pending_actions_multiple_actions(self):
        """Test extracting multiple pending actions."""
        class InterruptValue:
            def __init__(self):
                self.value = {
                    "action_requests": [
                        {"name": "action1", "args": {}},
                        {"name": "action2", "args": {}},
                        {"name": "action3", "args": {}}
                    ],
                    "review_configs": []
                }

        handler = ApprovalHandler()
        result = {"__interrupt__": [InterruptValue()]}
        actions = handler.extract_pending_actions(result)

        assert len(actions) == 3


class TestApprovalHandlerExtractReviewConfigs:
    """Tests for extract_review_configs method."""

    def test_extract_review_configs_success(self, sample_interrupt_result):
        """Test extracting review configs from interrupt."""
        handler = ApprovalHandler()
        configs = handler.extract_review_configs(sample_interrupt_result)

        assert len(configs) == 1
        assert configs[0]["action_name"] == "get_documents"
        assert "approve" in configs[0]["allowed_decisions"]

    def test_extract_review_configs_no_interrupt(self):
        """Test extracting configs when no interrupt."""
        handler = ApprovalHandler()
        result = {}
        configs = handler.extract_review_configs(result)

        assert configs == []


class TestApprovalHandlerFormatActionForDisplay:
    """Tests for format_action_for_display method."""

    def test_format_get_documents_action(self):
        """Test formatting get_documents action."""
        handler = ApprovalHandler()
        action = {
            "name": "get_documents",
            "args": {"document_ids": ["doc_001", "doc_002"]}
        }
        review_config = {
            "action_name": "get_documents",
            "allowed_decisions": ["approve", "reject", "edit"]
        }

        formatted = handler.format_action_for_display(action, review_config)

        assert "get_documents" in formatted
        assert "doc_001" in formatted
        assert "approve" in formatted
        assert "reject" in formatted
        assert "page-by-page summaries" in formatted

    def test_format_web_fetch_action(self):
        """Test formatting web_fetch action."""
        handler = ApprovalHandler()
        action = {
            "name": "web_fetch",
            "args": {"url": "https://example.com/legal"}
        }
        review_config = {
            "allowed_decisions": ["approve", "reject"]
        }

        formatted = handler.format_action_for_display(action, review_config)

        assert "web_fetch" in formatted
        assert "https://example.com/legal" in formatted
        assert "authoritative source" in formatted

    def test_format_write_file_action(self):
        """Test formatting write_file action."""
        handler = ApprovalHandler()
        action = {
            "name": "write_file",
            "args": {"path": "/output/report.txt", "content": "Report content"}
        }
        review_config = {
            "allowed_decisions": ["approve", "reject"]
        }

        formatted = handler.format_action_for_display(action, review_config)

        assert "write_file" in formatted
        assert "filesystem" in formatted

    def test_format_task_action(self):
        """Test formatting task action."""
        handler = ApprovalHandler()
        action = {
            "name": "task",
            "args": {"description": "Analyze corporate governance"}
        }
        review_config = {
            "allowed_decisions": ["approve", "reject"]
        }

        formatted = handler.format_action_for_display(action, review_config)

        assert "task" in formatted
        assert "subagent" in formatted

    def test_format_action_truncates_long_values(self):
        """Test that long argument values are truncated."""
        handler = ApprovalHandler()
        action = {
            "name": "test_action",
            "args": {"long_arg": "x" * 500}
        }
        review_config = {
            "allowed_decisions": ["approve"]
        }

        formatted = handler.format_action_for_display(action, review_config)

        # Should be truncated to ~200 chars + "..."
        assert "..." in formatted

    def test_format_action_no_args(self):
        """Test formatting action with no arguments."""
        handler = ApprovalHandler()
        action = {
            "name": "simple_action",
            "args": {}
        }
        review_config = {
            "allowed_decisions": ["approve"]
        }

        formatted = handler.format_action_for_display(action, review_config)

        assert "simple_action" in formatted

    def test_format_unknown_tool(self):
        """Test formatting unknown tool type."""
        handler = ApprovalHandler()
        action = {
            "name": "unknown_tool",
            "args": {"param": "value"}
        }
        review_config = {
            "allowed_decisions": ["approve"]
        }

        formatted = handler.format_action_for_display(action, review_config)

        assert "unknown_tool" in formatted
        assert "param" in formatted


class TestApprovalHandlerPromptForDecision:
    """Tests for prompt_for_decision method."""

    @patch('builtins.input')
    @patch('builtins.print')
    def test_prompt_approve_decision(self, mock_print, mock_input):
        """Test prompting and receiving approve decision."""
        handler = ApprovalHandler()
        action = {"name": "test_action", "args": {}}
        review_config = {
            "allowed_decisions": ["approve", "reject"]
        }

        mock_input.return_value = "1"  # Select approve

        decision = handler.prompt_for_decision(action, review_config)

        assert decision["type"] == "approve"

    @patch('builtins.input')
    @patch('builtins.print')
    def test_prompt_reject_decision(self, mock_print, mock_input):
        """Test prompting and receiving reject decision."""
        handler = ApprovalHandler()
        action = {"name": "test_action", "args": {}}
        review_config = {
            "allowed_decisions": ["approve", "reject"]
        }

        mock_input.return_value = "2"  # Select reject

        decision = handler.prompt_for_decision(action, review_config)

        assert decision["type"] == "reject"

    @patch('builtins.input')
    @patch('builtins.print')
    def test_prompt_invalid_then_valid(self, mock_print, mock_input):
        """Test handling invalid input then valid input."""
        handler = ApprovalHandler()
        action = {"name": "test_action", "args": {}}
        review_config = {
            "allowed_decisions": ["approve", "reject"]
        }

        # First invalid, then valid
        mock_input.side_effect = ["invalid", "5", "1"]

        decision = handler.prompt_for_decision(action, review_config)

        assert decision["type"] == "approve"

    @patch('builtins.input')
    @patch('builtins.print')
    def test_prompt_edit_decision_success(self, mock_print, mock_input):
        """Test edit decision with valid JSON."""
        handler = ApprovalHandler()
        action = {
            "name": "get_documents",
            "args": {"document_ids": ["doc_001"]}
        }
        review_config = {
            "allowed_decisions": ["approve", "reject", "edit"]
        }

        # Select edit, then provide JSON
        edited_json = '{"document_ids": ["doc_001", "doc_002"]}'
        mock_input.side_effect = ["3", edited_json]

        decision = handler.prompt_for_decision(action, review_config)

        assert decision["type"] == "edit"
        assert decision["edited_action"]["args"]["document_ids"] == ["doc_001", "doc_002"]

    @patch('builtins.input')
    @patch('builtins.print')
    def test_prompt_edit_decision_invalid_json(self, mock_print, mock_input):
        """Test edit decision with invalid JSON falls back to approve."""
        handler = ApprovalHandler()
        action = {
            "name": "get_documents",
            "args": {"document_ids": ["doc_001"]}
        }
        review_config = {
            "allowed_decisions": ["approve", "reject", "edit"]
        }

        # Select edit, then provide invalid JSON
        mock_input.side_effect = ["3", "not valid json"]

        decision = handler.prompt_for_decision(action, review_config)

        assert decision["type"] == "approve"


class TestApprovalHandlerProcessInterrupt:
    """Tests for process_interrupt method."""

    @patch('builtins.input')
    @patch('builtins.print')
    def test_process_interrupt_single_action(self, mock_print, mock_input, sample_interrupt_result):
        """Test processing interrupt with single action."""
        handler = ApprovalHandler()
        mock_input.return_value = "1"  # Approve

        decisions = handler.process_interrupt(sample_interrupt_result)

        assert len(decisions) == 1
        assert decisions[0]["type"] == "approve"

    @patch('builtins.input')
    @patch('builtins.print')
    def test_process_interrupt_multiple_actions(self, mock_print, mock_input):
        """Test processing interrupt with multiple actions."""
        class InterruptValue:
            def __init__(self):
                self.value = {
                    "action_requests": [
                        {"name": "action1", "args": {}},
                        {"name": "action2", "args": {}}
                    ],
                    "review_configs": [
                        {"action_name": "action1", "allowed_decisions": ["approve", "reject"]},
                        {"action_name": "action2", "allowed_decisions": ["approve", "reject"]}
                    ]
                }

        handler = ApprovalHandler()
        result = {"__interrupt__": [InterruptValue()]}

        # Approve first, reject second
        mock_input.side_effect = ["1", "2"]

        decisions = handler.process_interrupt(result)

        assert len(decisions) == 2
        assert decisions[0]["type"] == "approve"
        assert decisions[1]["type"] == "reject"

    def test_process_interrupt_no_actions(self):
        """Test processing interrupt with no actions."""
        handler = ApprovalHandler()
        result = {"messages": []}

        decisions = handler.process_interrupt(result)

        assert decisions == []


class TestApprovalHandlerCreateResumeCommand:
    """Tests for create_resume_command method."""

    def test_create_resume_command_single_decision(self):
        """Test creating resume command with single decision."""
        handler = ApprovalHandler()
        decisions = [{"type": "approve"}]

        command = handler.create_resume_command(decisions)

        assert isinstance(command, Command)
        assert command.resume == {"decisions": decisions}

    def test_create_resume_command_multiple_decisions(self):
        """Test creating resume command with multiple decisions."""
        handler = ApprovalHandler()
        decisions = [
            {"type": "approve"},
            {"type": "reject"},
            {"type": "edit", "edited_action": {"name": "test", "args": {}}}
        ]

        command = handler.create_resume_command(decisions)

        assert isinstance(command, Command)
        assert len(command.resume["decisions"]) == 3

    def test_create_resume_command_empty_decisions(self):
        """Test creating resume command with empty decisions."""
        handler = ApprovalHandler()
        decisions = []

        command = handler.create_resume_command(decisions)

        assert command.resume == {"decisions": []}


class TestGlobalApprovalHandler:
    """Tests for the global approval_handler instance."""

    def test_global_handler_exists(self):
        """Test that global approval_handler is instantiated."""
        assert approval_handler is not None
        assert isinstance(approval_handler, ApprovalHandler)

    def test_global_handler_methods(self):
        """Test that global handler has all required methods."""
        assert hasattr(approval_handler, 'check_for_interrupt')
        assert hasattr(approval_handler, 'extract_pending_actions')
        assert hasattr(approval_handler, 'extract_review_configs')
        assert hasattr(approval_handler, 'format_action_for_display')
        assert hasattr(approval_handler, 'prompt_for_decision')
        assert hasattr(approval_handler, 'process_interrupt')
        assert hasattr(approval_handler, 'create_resume_command')


class TestApprovalWorkflowIntegration:
    """Integration tests for approval workflow."""

    @patch('builtins.input')
    @patch('builtins.print')
    def test_complete_approval_flow(self, mock_print, mock_input):
        """Test complete approval workflow from interrupt to resume command."""
        handler = ApprovalHandler()

        # Create interrupt result
        class InterruptValue:
            def __init__(self):
                self.value = {
                    "action_requests": [
                        {
                            "name": "get_documents",
                            "args": {"document_ids": ["doc_001"]}
                        }
                    ],
                    "review_configs": [
                        {
                            "action_name": "get_documents",
                            "allowed_decisions": ["approve", "reject", "edit"]
                        }
                    ]
                }

        result = {"__interrupt__": [InterruptValue()]}

        # User approves the action
        mock_input.return_value = "1"

        # Check for interrupt
        assert handler.check_for_interrupt(result) is True

        # Process the interrupt
        decisions = handler.process_interrupt(result)
        assert len(decisions) == 1
        assert decisions[0]["type"] == "approve"

        # Create resume command
        command = handler.create_resume_command(decisions)
        assert isinstance(command, Command)
        assert command.resume["decisions"][0]["type"] == "approve"
