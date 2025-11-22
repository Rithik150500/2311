"""
test_main_application.py

Tests for the main application that orchestrates legal risk analysis workflow.
"""

import pytest
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunLegalRiskAnalysis:
    """Tests for run_legal_risk_analysis function."""

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_returns_result_dict(self, mock_storage, mock_approval, mock_agent):
        """Test that function returns result dictionary."""
        mock_storage.list_all_documents.return_value = []
        mock_agent.invoke.return_value = {"messages": []}
        mock_approval.check_for_interrupt.return_value = False

        from main_application import run_legal_risk_analysis
        result = run_legal_risk_analysis(
            data_room_path="./test_data_room",
            output_directory="./test_output"
        )

        assert "session_id" in result
        assert "status" in result
        assert "iterations" in result
        assert "final_result" in result

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_creates_output_directory(self, mock_storage, mock_approval, mock_agent, tmp_path):
        """Test that output directory is created."""
        mock_storage.list_all_documents.return_value = []
        mock_agent.invoke.return_value = {"messages": []}
        mock_approval.check_for_interrupt.return_value = False

        output_dir = tmp_path / "new_output"

        from main_application import run_legal_risk_analysis
        run_legal_risk_analysis(
            data_room_path="./test_data_room",
            output_directory=str(output_dir)
        )

        assert output_dir.exists()

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_generates_unique_thread_id(self, mock_storage, mock_approval, mock_agent):
        """Test that unique thread ID is generated."""
        mock_storage.list_all_documents.return_value = []
        mock_agent.invoke.return_value = {"messages": []}
        mock_approval.check_for_interrupt.return_value = False

        from main_application import run_legal_risk_analysis
        result1 = run_legal_risk_analysis("./test", "./output1")
        result2 = run_legal_risk_analysis("./test", "./output2")

        assert result1["session_id"] != result2["session_id"]
        # Verify it's a valid UUID
        uuid.UUID(result1["session_id"])

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_loads_document_summaries(self, mock_storage, mock_approval, mock_agent):
        """Test that document summaries are loaded from storage."""
        mock_storage.list_all_documents.return_value = [
            {
                "id": "doc_001",
                "title": "Test Doc",
                "document_type": "Contract",
                "page_count": 5,
                "summary_description": "A test document"
            }
        ]
        mock_agent.invoke.return_value = {"messages": []}
        mock_approval.check_for_interrupt.return_value = False

        from main_application import run_legal_risk_analysis
        run_legal_risk_analysis("./test", "./output")

        mock_storage.list_all_documents.assert_called_once()

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_invokes_agent_with_correct_config(self, mock_storage, mock_approval, mock_agent):
        """Test that agent is invoked with correct configuration."""
        mock_storage.list_all_documents.return_value = []
        mock_agent.invoke.return_value = {"messages": []}
        mock_approval.check_for_interrupt.return_value = False

        from main_application import run_legal_risk_analysis
        run_legal_risk_analysis("./test", "./output")

        # Check agent was invoked
        mock_agent.invoke.assert_called()
        call_args = mock_agent.invoke.call_args
        config = call_args.kwargs.get('config') or call_args[1].get('config')
        assert "configurable" in config
        assert "thread_id" in config["configurable"]

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_initial_message_contains_documents(self, mock_storage, mock_approval, mock_agent):
        """Test that initial message contains document information."""
        mock_storage.list_all_documents.return_value = [
            {
                "id": "doc_001",
                "title": "Test Document",
                "document_type": "Contract",
                "page_count": 5,
                "summary_description": "Summary text"
            }
        ]
        mock_agent.invoke.return_value = {"messages": []}
        mock_approval.check_for_interrupt.return_value = False

        from main_application import run_legal_risk_analysis
        run_legal_risk_analysis("./test", "./output")

        call_args = mock_agent.invoke.call_args
        messages = call_args[0][0]["messages"]
        content = messages[0]["content"]

        assert "Test Document" in content
        assert "doc_001" in content
        assert "Contract" in content


class TestApprovalLoop:
    """Tests for the approval loop behavior."""

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_no_interrupt_completes_immediately(self, mock_storage, mock_approval, mock_agent):
        """Test that execution completes when no interrupt occurs."""
        mock_storage.list_all_documents.return_value = []
        mock_agent.invoke.return_value = {"messages": []}
        mock_approval.check_for_interrupt.return_value = False

        from main_application import run_legal_risk_analysis
        result = run_legal_risk_analysis("./test", "./output")

        assert result["status"] == "complete"
        assert result["iterations"] == 1

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_processes_single_interrupt(self, mock_storage, mock_approval, mock_agent):
        """Test processing a single interrupt."""
        mock_storage.list_all_documents.return_value = []

        # First invoke returns interrupt, second returns complete
        mock_agent.invoke.side_effect = [
            {"__interrupt__": True, "messages": []},
            {"messages": []}
        ]
        mock_approval.check_for_interrupt.side_effect = [True, False]
        mock_approval.process_interrupt.return_value = [{"type": "approve"}]
        mock_approval.create_resume_command.return_value = MagicMock()

        from main_application import run_legal_risk_analysis
        result = run_legal_risk_analysis("./test", "./output")

        assert result["iterations"] == 2
        mock_approval.process_interrupt.assert_called_once()

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_processes_multiple_interrupts(self, mock_storage, mock_approval, mock_agent):
        """Test processing multiple interrupts."""
        mock_storage.list_all_documents.return_value = []

        # Three invokes: two interrupts, then complete
        mock_agent.invoke.side_effect = [
            {"__interrupt__": True, "messages": []},
            {"__interrupt__": True, "messages": []},
            {"messages": []}
        ]
        mock_approval.check_for_interrupt.side_effect = [True, True, False]
        mock_approval.process_interrupt.return_value = [{"type": "approve"}]
        mock_approval.create_resume_command.return_value = MagicMock()

        from main_application import run_legal_risk_analysis
        result = run_legal_risk_analysis("./test", "./output")

        assert result["iterations"] == 3
        assert mock_approval.process_interrupt.call_count == 2

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_respects_max_iterations(self, mock_storage, mock_approval, mock_agent):
        """Test that max iterations limit is enforced."""
        mock_storage.list_all_documents.return_value = []

        # Always return interrupt
        mock_agent.invoke.return_value = {"__interrupt__": True, "messages": []}
        mock_approval.check_for_interrupt.return_value = True
        mock_approval.process_interrupt.return_value = [{"type": "approve"}]
        mock_approval.create_resume_command.return_value = MagicMock()

        from main_application import run_legal_risk_analysis
        result = run_legal_risk_analysis("./test", "./output")

        assert result["iterations"] == 50  # Max iterations

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_empty_decisions_aborts(self, mock_storage, mock_approval, mock_agent):
        """Test that empty decisions aborts the loop."""
        mock_storage.list_all_documents.return_value = []

        mock_agent.invoke.return_value = {"__interrupt__": True, "messages": []}
        mock_approval.check_for_interrupt.return_value = True
        mock_approval.process_interrupt.return_value = []  # Empty decisions

        from main_application import run_legal_risk_analysis
        result = run_legal_risk_analysis("./test", "./output")

        assert result["iterations"] == 1  # Should abort after first interrupt


class TestResumeExecution:
    """Tests for resume execution behavior."""

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_resume_with_correct_command(self, mock_storage, mock_approval, mock_agent):
        """Test that agent is resumed with correct command."""
        mock_storage.list_all_documents.return_value = []

        mock_resume_command = MagicMock()
        mock_agent.invoke.side_effect = [
            {"__interrupt__": True, "messages": []},
            {"messages": []}
        ]
        mock_approval.check_for_interrupt.side_effect = [True, False]
        mock_approval.process_interrupt.return_value = [{"type": "approve"}]
        mock_approval.create_resume_command.return_value = mock_resume_command

        from main_application import run_legal_risk_analysis
        run_legal_risk_analysis("./test", "./output")

        # Check resume was called with the command
        second_call = mock_agent.invoke.call_args_list[1]
        assert second_call[0][0] == mock_resume_command

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_resume_maintains_config(self, mock_storage, mock_approval, mock_agent):
        """Test that config is maintained during resume."""
        mock_storage.list_all_documents.return_value = []

        mock_agent.invoke.side_effect = [
            {"__interrupt__": True, "messages": []},
            {"messages": []}
        ]
        mock_approval.check_for_interrupt.side_effect = [True, False]
        mock_approval.process_interrupt.return_value = [{"type": "approve"}]
        mock_approval.create_resume_command.return_value = MagicMock()

        from main_application import run_legal_risk_analysis
        run_legal_risk_analysis("./test", "./output")

        # Both calls should have the same thread_id in config
        first_config = mock_agent.invoke.call_args_list[0][1]["config"]
        second_config = mock_agent.invoke.call_args_list[1][1]["config"]
        assert first_config["configurable"]["thread_id"] == second_config["configurable"]["thread_id"]


class TestResultExtraction:
    """Tests for result extraction and formatting."""

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_final_result_includes_messages(self, mock_storage, mock_approval, mock_agent):
        """Test that final result includes agent messages."""
        mock_storage.list_all_documents.return_value = []
        mock_agent.invoke.return_value = {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"}
            ]
        }
        mock_approval.check_for_interrupt.return_value = False

        from main_application import run_legal_risk_analysis
        result = run_legal_risk_analysis("./test", "./output")

        assert "messages" in result["final_result"]
        assert len(result["final_result"]["messages"]) == 2

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_status_is_complete(self, mock_storage, mock_approval, mock_agent):
        """Test that status is complete on successful execution."""
        mock_storage.list_all_documents.return_value = []
        mock_agent.invoke.return_value = {"messages": []}
        mock_approval.check_for_interrupt.return_value = False

        from main_application import run_legal_risk_analysis
        result = run_legal_risk_analysis("./test", "./output")

        assert result["status"] == "complete"


class TestOutputFormatting:
    """Tests for output formatting and printing."""

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    @patch('builtins.print')
    def test_prints_session_info(self, mock_print, mock_storage, mock_approval, mock_agent):
        """Test that session information is printed."""
        mock_storage.list_all_documents.return_value = []
        mock_agent.invoke.return_value = {"messages": []}
        mock_approval.check_for_interrupt.return_value = False

        from main_application import run_legal_risk_analysis
        result = run_legal_risk_analysis("./test", "./output")

        # Verify session ID was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        session_printed = any(result["session_id"] in str(call) for call in print_calls)
        assert session_printed

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    @patch('builtins.print')
    def test_prints_completion_message(self, mock_print, mock_storage, mock_approval, mock_agent):
        """Test that completion message is printed."""
        mock_storage.list_all_documents.return_value = []
        mock_agent.invoke.return_value = {"messages": []}
        mock_approval.check_for_interrupt.return_value = False

        from main_application import run_legal_risk_analysis
        run_legal_risk_analysis("./test", "./output")

        print_calls = [str(call) for call in mock_print.call_args_list]
        complete_printed = any("COMPLETE" in str(call) or "complete" in str(call) for call in print_calls)
        assert complete_printed


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_handles_empty_document_list(self, mock_storage, mock_approval, mock_agent):
        """Test handling of empty document list."""
        mock_storage.list_all_documents.return_value = []
        mock_agent.invoke.return_value = {"messages": []}
        mock_approval.check_for_interrupt.return_value = False

        from main_application import run_legal_risk_analysis
        result = run_legal_risk_analysis("./test", "./output")

        # Should still complete successfully
        assert result["status"] == "complete"

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_handles_missing_messages_in_result(self, mock_storage, mock_approval, mock_agent):
        """Test handling when messages are missing from result."""
        mock_storage.list_all_documents.return_value = []
        mock_agent.invoke.return_value = {}  # No messages key
        mock_approval.check_for_interrupt.return_value = False

        from main_application import run_legal_risk_analysis
        result = run_legal_risk_analysis("./test", "./output")

        # Should handle gracefully
        assert result["status"] == "complete"


class TestIntegration:
    """Integration tests for the main application."""

    @patch('main_application.legal_analysis_agent')
    @patch('main_application.approval_handler')
    @patch('main_application.data_room_storage')
    def test_complete_workflow_with_approvals(self, mock_storage, mock_approval, mock_agent, tmp_path):
        """Test complete workflow with multiple approvals."""
        mock_storage.list_all_documents.return_value = [
            {
                "id": "doc_001",
                "title": "Test Document",
                "document_type": "Contract",
                "page_count": 10,
                "summary_description": "A test contract document"
            }
        ]

        # Simulate 3 interrupts, then complete
        mock_agent.invoke.side_effect = [
            {"__interrupt__": True, "messages": [{"role": "assistant", "content": "Planning..."}]},
            {"__interrupt__": True, "messages": [{"role": "assistant", "content": "Analyzing..."}]},
            {"__interrupt__": True, "messages": [{"role": "assistant", "content": "Writing..."}]},
            {"messages": [{"role": "assistant", "content": "Analysis complete."}]}
        ]
        mock_approval.check_for_interrupt.side_effect = [True, True, True, False]
        mock_approval.process_interrupt.return_value = [{"type": "approve"}]
        mock_approval.create_resume_command.return_value = MagicMock()

        from main_application import run_legal_risk_analysis
        result = run_legal_risk_analysis(
            data_room_path="./test_data_room",
            output_directory=str(tmp_path / "output")
        )

        assert result["status"] == "complete"
        assert result["iterations"] == 4
        assert mock_approval.process_interrupt.call_count == 3
