"""Interactive ipywidgets for plan approval, clarification, and progress.

All functions detect ipywidgets at import time and are only called when
HAS_WIDGETS is True.  The background-thread ↔ main-kernel-thread bridge
uses threading.Event so the WS event loop can block until widget callbacks fire.
"""
from __future__ import annotations

import threading
import time
from typing import Any


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------
try:
    import ipywidgets as W  # noqa: N812
    from IPython.display import display

    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False


# ---------------------------------------------------------------------------
# Bridge: background thread ↔ widget callback (main kernel thread)
# ---------------------------------------------------------------------------
class _WidgetBridge:
    """Block a background thread until a widget callback fires."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: Any = None

    def wait(self, timeout: float | None = None) -> Any:
        self._event.wait(timeout=timeout)
        return self._result

    def resolve(self, result: Any) -> None:
        self._result = result
        self._event.set()


# ---------------------------------------------------------------------------
# Plan approval widget
# ---------------------------------------------------------------------------
def widget_plan_approval(data: dict) -> dict:
    """Display interactive plan approval widget, block until user decides.

    Returns dict compatible with existing approve/reject protocol:
      {"approved": True, "deleted_steps": [...], "edited_steps": [...]}
      {"approved": False, "feedback": "..."}
    """
    plan = data.get("plan", {})
    if plan and plan.get("steps"):
        steps = plan["steps"]
        problem = plan.get("problem", "")
    else:
        steps = data.get("steps", [])
        problem = data.get("problem", "")

    if not steps:
        return {"approved": False, "feedback": "Empty plan"}

    bridge = _WidgetBridge()

    # --- Build widgets ---
    title = W.HTML(value=f"<h3>Plan: {problem}</h3>")

    checkboxes: list[W.Checkbox] = []
    goal_labels: list[W.Label] = []
    edit_texts: list[W.Text] = []
    edit_buttons: list[W.Button] = []
    step_rows: list[W.HBox] = []

    for s in steps:
        num = s.get("number", "?")
        goal = s.get("goal", "")

        cb = W.Checkbox(value=True, indent=False,
                        layout=W.Layout(width="30px"))
        label = W.Label(value=f"{num}. {goal}",
                        layout=W.Layout(width="500px"))
        edit_text = W.Text(value=goal, layout=W.Layout(width="500px", display="none"))
        edit_btn = W.Button(description="✏", layout=W.Layout(width="32px"))

        def _toggle_edit(btn, _label=label, _text=edit_text):
            if _text.layout.display == "none":
                _text.layout.display = ""
                _label.layout.display = "none"
            else:
                _label.value = f"{_label.value.split('.')[0]}. {_text.value}"
                _text.layout.display = "none"
                _label.layout.display = ""

        edit_btn.on_click(_toggle_edit)

        checkboxes.append(cb)
        goal_labels.append(label)
        edit_texts.append(edit_text)
        edit_buttons.append(edit_btn)
        step_rows.append(W.HBox([cb, label, edit_text, edit_btn]))

    feedback_input = W.Text(
        placeholder="Feedback (optional, used on reject)",
        layout=W.Layout(width="540px"),
    )
    approve_btn = W.Button(description="Approve", button_style="success")
    reject_btn = W.Button(description="Reject", button_style="danger")

    def _on_approve(_btn):
        deleted = []
        edited = []
        for i, s in enumerate(steps):
            if not checkboxes[i].value:
                deleted.append(s.get("number", i + 1))
            elif edit_texts[i].value != s.get("goal", ""):
                edited.append({
                    "step_number": s.get("number", i + 1),
                    "goal": edit_texts[i].value,
                })
        result = {"approved": True}
        if deleted:
            result["deleted_steps"] = deleted
        if edited:
            result["edited_steps"] = edited
        approve_btn.disabled = True
        reject_btn.disabled = True
        bridge.resolve(result)

    def _on_reject(_btn):
        fb = feedback_input.value.strip() or "Rejected by user"
        approve_btn.disabled = True
        reject_btn.disabled = True
        bridge.resolve({"approved": False, "feedback": fb})

    approve_btn.on_click(_on_approve)
    reject_btn.on_click(_on_reject)

    box = W.VBox([
        title,
        *step_rows,
        W.HBox([W.Label("Feedback:"), feedback_input]),
        W.HBox([approve_btn, reject_btn]),
    ])

    display(box)
    return bridge.wait()


# ---------------------------------------------------------------------------
# Clarification widget
# ---------------------------------------------------------------------------
def widget_clarification(data: dict) -> dict[str, str] | None:
    """Display interactive clarification widget, block until user submits.

    Returns {question_text: answer_text} dict, or None if no questions.
    """
    questions = data.get("questions", [])
    if not questions:
        return None

    bridge = _WidgetBridge()

    reason = data.get("ambiguity_reason", "Clarification needed")
    header = W.HTML(value=f"<h3>Clarification: {reason}</h3>")

    question_widgets: list[tuple[str, Any]] = []

    for q in questions:
        text = q.get("text", "")
        suggestions = q.get("suggestions", [])
        q_label = W.HTML(value=f"<b>{text}</b>")

        if suggestions:
            options = suggestions + ["Custom"]
            radio = W.RadioButtons(options=options, layout=W.Layout(width="auto"))
            custom_text = W.Text(
                placeholder="Custom answer...",
                layout=W.Layout(width="400px", display="none"),
            )

            def _on_radio_change(change, _radio=radio, _custom=custom_text):
                if change["new"] == "Custom":
                    _custom.layout.display = ""
                else:
                    _custom.layout.display = "none"

            radio.observe(_on_radio_change, names="value")
            widget = W.VBox([q_label, radio, custom_text])
            question_widgets.append((text, ("radio", radio, custom_text)))
        else:
            text_input = W.Text(
                placeholder="Your answer...",
                layout=W.Layout(width="400px"),
            )
            widget = W.VBox([q_label, text_input])
            question_widgets.append((text, ("text", text_input)))

    submit_btn = W.Button(description="Submit", button_style="primary")

    def _on_submit(_btn):
        answers = {}
        for q_text, spec in question_widgets:
            if spec[0] == "radio":
                _, radio, custom = spec
                if radio.value == "Custom":
                    answers[q_text] = custom.value
                else:
                    answers[q_text] = radio.value
            else:
                _, text_input = spec
                answers[q_text] = text_input.value
        submit_btn.disabled = True
        bridge.resolve(answers)

    submit_btn.on_click(_on_submit)

    all_widgets = [header]
    for q_text, spec in question_widgets:
        if spec[0] == "radio":
            q_label = W.HTML(value=f"<b>{q_text}</b>")
            all_widgets.append(W.VBox([q_label, spec[1], spec[2]]))
        else:
            q_label = W.HTML(value=f"<b>{q_text}</b>")
            all_widgets.append(W.VBox([q_label, spec[1]]))
    all_widgets.append(submit_btn)

    display(W.VBox(all_widgets))
    return bridge.wait()


# ---------------------------------------------------------------------------
# Progress widget (replaces PrintProgress during execution)
# ---------------------------------------------------------------------------
class WidgetProgress:
    """ipywidgets-based progress display.

    Displayed once from the background thread via display(); subsequent
    updates are thread-safe property assignments.
    """

    def __init__(self) -> None:
        self._total_steps = 0
        self._current_step = 0
        self._step_goals: list[str] = []
        self._step_statuses: list[str] = []  # "pending" | "running" | "done" | "error"
        self._step_durations: list[float | None] = []
        self._displayed = False

        self._status_label = W.HTML(value="<b>Waiting...</b>")
        self._progress_bar = W.IntProgress(
            value=0, min=0, max=1, bar_style="info",
            layout=W.Layout(width="100%"),
        )
        self._step_list = W.HTML(value="")
        self._box = W.VBox([
            self._status_label,
            self._progress_bar,
            self._step_list,
        ])

    def _ensure_displayed(self) -> None:
        if not self._displayed:
            display(self._box)
            self._displayed = True

    def _render_steps(self) -> None:
        lines = []
        for i, goal in enumerate(self._step_goals):
            status = self._step_statuses[i]
            dur = self._step_durations[i]
            num = i + 1
            if status == "done":
                dur_str = f" ({dur / 1000:.1f}s)" if dur else ""
                lines.append(f"&nbsp;&nbsp;✓ Step {num}: {goal}{dur_str}")
            elif status == "running":
                lines.append(f"&nbsp;&nbsp;⟳ Step {num}: {goal}...")
            elif status == "error":
                lines.append(f"&nbsp;&nbsp;✗ Step {num}: {goal}")
            else:
                lines.append(f"&nbsp;&nbsp;○ Step {num}: {goal}")
        self._step_list.value = "<br>".join(lines)

    def handle_event(self, event_type: str, data: dict) -> None:
        self._ensure_displayed()

        match event_type:
            case "planning_start":
                self._status_label.value = "<b>Generating plan...</b>"

            case "plan_ready":
                plan = data.get("plan", {})
                if plan and plan.get("steps"):
                    steps = plan["steps"]
                    problem = plan.get("problem", "")
                else:
                    steps = data.get("steps", [])
                    problem = data.get("problem", "")

                self._total_steps = len(steps)
                self._step_goals = [s.get("goal", "") for s in steps]
                self._step_statuses = ["pending"] * self._total_steps
                self._step_durations = [None] * self._total_steps
                self._progress_bar.max = max(self._total_steps, 1)
                self._progress_bar.value = 0
                self._status_label.value = f"<b>Plan: {problem} ({self._total_steps} steps)</b>"
                self._render_steps()

            case "step_start":
                n = data.get("step_number", 0)
                goal = data.get("goal", "")
                self._current_step = n
                idx = n - 1
                if 0 <= idx < self._total_steps:
                    self._step_statuses[idx] = "running"
                self._status_label.value = f"<b>Step {n}/{self._total_steps}: {goal}</b>"
                self._render_steps()

            case "step_generating" | "step_executing":
                pass

            case "step_complete":
                n = data.get("step_number", self._current_step)
                idx = n - 1
                if 0 <= idx < self._total_steps:
                    self._step_statuses[idx] = "done"
                    self._step_durations[idx] = data.get("duration_ms")
                self._progress_bar.value = sum(
                    1 for s in self._step_statuses if s == "done"
                )
                self._render_steps()

            case "step_error" | "step_failed":
                idx = self._current_step - 1
                if 0 <= idx < self._total_steps:
                    self._step_statuses[idx] = "error"
                self._render_steps()

            case "synthesizing":
                self._status_label.value = "<b>Generating answer...</b>"

            case "proof_start":
                self._status_label.value = "<b>Reasoning chain starting...</b>"

            case "fact_start":
                name = data.get("name", "")
                self._status_label.value = f"<b>Resolving premise: {name}...</b>"

            case "fact_resolved":
                name = data.get("name", "")
                self._status_label.value = f"<b>Premise resolved: {name}</b>"

            case "fact_failed":
                name = data.get("name", "")
                self._status_label.value = f"<b>Premise FAILED: {name}</b>"

            case "dag_execution_start":
                self._status_label.value = "<b>Executing inferences...</b>"

            case "inference_executing":
                name = data.get("name", "")
                self._status_label.value = f"<b>Inference: {name}...</b>"

            case "inference_complete":
                name = data.get("name", "")
                self._status_label.value = f"<b>Inference complete: {name}</b>"

            case "proof_complete":
                self._status_label.value = "<b>Reasoning complete</b>"

            case "query_complete":
                self._progress_bar.value = self._progress_bar.max
                self._progress_bar.bar_style = "success"
                self._status_label.value = "<b>Done</b>"
                # Mark any remaining as done
                for i, s in enumerate(self._step_statuses):
                    if s == "running":
                        self._step_statuses[i] = "done"
                self._render_steps()

            case "query_error":
                error = data.get("error", "")
                first_line = error.split("\n")[0][:120] if error else "Unknown error"
                self._progress_bar.bar_style = "danger"
                self._status_label.value = f"<b>Error: {first_line}</b>"

            case "clarification_needed":
                pass  # Handled by widget_clarification
