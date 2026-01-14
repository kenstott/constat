"""Tests for multi-step Session execution.

Requires ANTHROPIC_API_KEY to be set.
"""

import os
import pytest
import tempfile
from pathlib import Path

from constat.core.config import Config
from constat.core.models import ArtifactType
from constat.storage.history import SessionHistory
from constat.session import Session, SessionConfig
from constat.catalog.schema_manager import SchemaManager


pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)

FIXTURES_DIR = Path(__file__).parent.parent
CHINOOK_DB = FIXTURES_DIR / "data" / "chinook.db"
NORTHWIND_DB = FIXTURES_DIR / "data" / "northwind.db"


@pytest.fixture(scope="module")
def temp_history_dir():
    """Create temp directory for session history (module-scoped for shared tests)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "sessions"


@pytest.fixture
def fresh_history_dir():
    """Create a fresh temp directory for each test (function-scoped)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "sessions"


@pytest.fixture(scope="module")
def session(temp_history_dir) -> Session:
    """Create a session with both databases."""
    config = Config(
        databases={
            "chinook": {"uri": f"sqlite:///{CHINOOK_DB}"},
            "northwind": {"uri": f"sqlite:///{NORTHWIND_DB}"},
        },
        system_prompt="""You are analyzing data across two business databases.

## Chinook (Digital Music Store)
- Artists, Albums, Tracks, Genres
- Customers purchase via Invoices/InvoiceLines
- Each database represents a single company

## Northwind (Product Distribution)
- Products, Categories, Suppliers
- Customers place Orders with Order Details
- Each database represents a single company
""",
        execution={"allowed_imports": ["pandas", "numpy", "json", "datetime"]},
    )

    history = SessionHistory(storage_dir=temp_history_dir)
    session_config = SessionConfig(max_retries_per_step=3)

    return Session(config, session_config=session_config, history=history)


class TestMultiStepSession:
    """Test multi-step plan execution."""

    def test_simple_multi_step_query(self, session: Session):
        """Test a simple multi-step query."""
        result = session.solve(
            "List the top 3 music genres by number of tracks in Chinook. "
            "Then analyze what percentage of total tracks each represents."
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        assert result["plan"] is not None
        assert len(result["plan"].steps) >= 2  # At least 2 steps

        print(f"\n--- Multi-Step Query ---")
        print(f"Plan: {len(result['plan'].steps)} steps")
        for step in result["plan"].steps:
            print(f"  Step {step.number}: {step.goal}")
        print(f"\nOutput:\n{result['output'][:500]}")

    def test_cross_database_multi_step(self, session: Session):
        """Test multi-step query across databases."""
        result = session.solve(
            "Compare employee counts between Chinook and Northwind. "
            "Which company has more employees? "
            "Then list the names of all employees from the larger company."
        )

        assert result["success"], f"Query failed: {result.get('error')}"

        # Should reference both databases in the plan
        plan_text = " ".join(step.goal for step in result["plan"].steps).lower()
        has_both = "chinook" in plan_text or "northwind" in plan_text

        print(f"\n--- Cross-DB Multi-Step Query ---")
        print(f"Plan: {len(result['plan'].steps)} steps")
        print(f"\nOutput:\n{result['output']}")

    def test_state_sharing_between_steps(self, session: Session):
        """Test that state is properly shared between steps."""
        result = session.solve(
            "Analyze the Chinook database: first, count the total number of tracks. "
            "Then, find how many tracks are in the 'Rock' genre. "
            "Finally, summarize what percentage of tracks are Rock."
        )

        assert result["success"], f"Query failed: {result.get('error')}"

        # The final step should use results from previous steps
        output = result["output"].lower()
        assert "rock" in output
        assert "%" in output or "percent" in output

        print(f"\n--- State Sharing Test ---")
        if "scratchpad" in result:
            print(f"Scratchpad:\n{result['scratchpad'][:1000]}")
        else:
            print(f"Output:\n{result['output'][:1000]}")


class TestPlanner:
    """Test the planning phase."""

    def test_plan_generation(self, session: Session):
        """Test that plans are generated correctly."""
        planner_response = session.planner.plan(
            "Analyze the top 5 customers by total purchase amount in Chinook"
        )

        plan = planner_response.plan
        assert len(plan.steps) >= 1
        assert plan.problem == "Analyze the top 5 customers by total purchase amount in Chinook"

        # Steps should have goals
        for step in plan.steps:
            assert step.goal
            assert step.number > 0

        print(f"\n--- Plan Generation ---")
        print(f"Reasoning: {planner_response.reasoning}")
        for step in plan.steps:
            print(f"  Step {step.number}: {step.goal}")
            print(f"    Inputs: {step.expected_inputs}")
            print(f"    Outputs: {step.expected_outputs}")


class TestSessionHistory:
    """Test session history integration."""

    def test_session_recorded(self, session: Session, temp_history_dir):
        """Test that session is recorded in history."""
        # Run a query
        result = session.solve("Count the number of artists in Chinook")

        assert result["success"]
        assert session.session_id is not None

        # Check history
        history = SessionHistory(storage_dir=temp_history_dir)
        sessions = history.list_sessions()

        assert len(sessions) > 0
        # Our session should be there
        session_ids = [s.session_id for s in sessions]
        assert session.session_id in session_ids


class TestEventHandling:
    """Test event emission during execution."""

    def test_events_emitted(self, session: Session):
        """Test that events are emitted during execution."""
        events = []

        def capture_event(event):
            events.append(event)

        session.on_event(capture_event)

        result = session.solve("Show me how many genres are in Chinook")

        assert result["success"]
        assert len(events) > 0

        # Check event types
        event_types = [e.event_type for e in events]
        assert "step_start" in event_types
        assert "generating" in event_types
        assert "executing" in event_types
        assert "step_complete" in event_types

        print(f"\n--- Events ---")
        for event in events:
            print(f"  {event.event_type}: step {event.step_number}")


class TestSessionResumption:
    """Test session resumption and follow-up queries."""

    def test_follow_up_has_context(self, fresh_history_dir):
        """
        Test that follow-up queries have access to data from previous steps.

        This test:
        1. Runs an initial query that creates a table
        2. Creates a NEW session instance (simulating app restart)
        3. Resumes the previous session
        4. Asks a follow-up that references the previous data
        5. Verifies the follow-up can access that context
        """
        # Create initial session
        config = Config(
            databases={
                "chinook": {"uri": f"sqlite:///{CHINOOK_DB}"},
            },
            system_prompt="You are analyzing a music store database.",
            execution={"allowed_imports": ["pandas", "numpy", "json", "datetime"]},
        )

        history = SessionHistory(storage_dir=fresh_history_dir)
        session1 = Session(config, session_config=SessionConfig(max_retries_per_step=3), history=history)

        # Step 1: Initial query - get top genres and save to table
        print("\n--- Step 1: Initial Query ---")
        result1 = session1.solve(
            "Get the top 5 genres by track count from Chinook. "
            "Save the result as a table called 'top_genres'."
        )

        assert result1["success"], f"Initial query failed: {result1.get('error')}"
        session_id = session1.session_id
        print(f"Session ID: {session_id}")
        print(f"Tables created: {result1.get('datastore_tables', [])}")
        print(f"Output:\n{result1['output'][:500]}")

        # Verify table was created
        tables_after_step1 = [t["name"] for t in result1.get("datastore_tables", [])]
        print(f"Tables after step 1: {tables_after_step1}")

        # Step 2: Create a NEW session instance (simulating app restart)
        print("\n--- Step 2: Resume Session ---")
        session2 = Session(config, session_config=SessionConfig(max_retries_per_step=3), history=history)

        # Resume the previous session
        resumed = session2.resume(session_id)
        assert resumed, f"Failed to resume session {session_id}"
        print(f"Resumed session: {session2.session_id}")

        # Check that the datastore has the previous tables
        tables_after_resume = session2.datastore.list_tables()
        print(f"Tables after resume: {[t['name'] for t in tables_after_resume]}")

        # Step 3: Follow-up query that uses data from step 1
        print("\n--- Step 3: Follow-up Query ---")
        result2 = session2.follow_up(
            "Using the top_genres data from the previous step, "
            "what is the #1 genre and how many tracks does it have?"
        )

        assert result2["success"], f"Follow-up query failed: {result2.get('error')}"
        print(f"Follow-up output:\n{result2['output']}")

        # Verify the follow-up actually used the context
        output_lower = result2["output"].lower()
        # The output should mention the #1 genre (likely Rock or similar)
        assert any(genre in output_lower for genre in ["rock", "latin", "metal", "jazz", "alternative"]), \
            "Follow-up should mention a genre from the previous data"

        # Check scratchpad shows continuity (step numbers should continue)
        print(f"\nFull scratchpad:\n{result2['scratchpad']}")

        # Verify artifacts (code) were saved for each step
        all_artifacts = session2.datastore.get_artifacts()
        print(f"\n--- Saved Artifacts ---")
        for artifact in all_artifacts:
            print(f"  Step {artifact.step_number}, Attempt {artifact.attempt}, Type: {artifact.artifact_type}")
            if artifact.artifact_type == ArtifactType.CODE:
                print(f"    Code preview: {artifact.content[:100]}...")

        # Should have code artifacts for multiple steps
        code_artifacts = [a for a in all_artifacts if a.artifact_type == ArtifactType.CODE]
        assert len(code_artifacts) >= 2, "Should have code artifacts from both initial and follow-up"

    def test_resume_nonexistent_session(self, fresh_history_dir):
        """Test that resuming a non-existent session returns False."""
        config = Config(
            databases={"chinook": {"uri": f"sqlite:///{CHINOOK_DB}"}},
        )
        history = SessionHistory(storage_dir=fresh_history_dir)
        session = Session(config, history=history)

        resumed = session.resume("nonexistent-session-id")
        assert not resumed

    def test_follow_up_without_active_session_raises(self, fresh_history_dir):
        """Test that follow_up without an active session raises an error."""
        config = Config(
            databases={"chinook": {"uri": f"sqlite:///{CHINOOK_DB}"}},
        )
        history = SessionHistory(storage_dir=fresh_history_dir)
        session = Session(config, history=history)

        with pytest.raises(ValueError, match="No active session"):
            session.follow_up("Some follow-up question")
