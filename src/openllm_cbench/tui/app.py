"""
`cbench tui` -- a thin Textual control panel over the finished CLI.

Every screen here does one of two things: (1) shell out to a real
`cbench` subcommand via jobs.run_job() and stream its output, or (2)
read files `cbench` itself already wrote (results/, the model catalogue).
There is no suite logic in this file -- see jobs.py's module docstring
for why that split is load-bearing, not just tidy.
"""

import os
import sys
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button, Checkbox, DirectoryTree, Footer, Header, Input, RichLog,
    Select, Static, TextArea,
)

from openllm_cbench.core.invariant import SAFETY_INVARIANT
from openllm_cbench.tui.jobs import RUNNABLE_SUITES, build_args, cbench_command, run_job


def _results_root() -> Path:
    # Same precedence core.paths.results_dir() uses for a single suite,
    # applied to the shared results/ root the TUI's report browser lists.
    override = os.environ.get("OPENLLM_CBENCH_RESULTS_DIR")
    return Path(override) if override else Path.cwd() / "results"


class InvariantBar(Static):
    """The safety invariant, visible on every screen -- every CLI --help
    epilog already carries it, so the TUI gets the same treatment rather
    than being the one surface it's missing from."""

    def compose(self) -> ComposeResult:
        yield Static(f"[bold]{SAFETY_INVARIANT}[/bold]", id="invariant-text")


class DashboardScreen(Screen):
    BINDINGS = [("d", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        with Vertical(id="dashboard-body"):
            yield Static("openllm-cbench", id="title")
            with Horizontal(id="dashboard-buttons"):
                yield Button("Run a suite", id="goto-run", variant="primary")
                yield Button("Gate a model", id="goto-gate")
                yield Button("Browse reports", id="goto-reports")
                yield Button("Refresh doctor", id="run-doctor")
            yield RichLog(id="doctor-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        # Deliberately does NOT run `cbench doctor` automatically -- every
        # cbench action this TUI takes happens because of a click, none on
        # app startup, so what ran and why is never ambiguous. Press
        # "Refresh doctor" to check the environment.
        self.query_one("#doctor-log", RichLog).write(
            "[dim]Press \"Refresh doctor\" to check your environment "
            "(endpoint, canary self-check, hardware, catalogue).[/dim]"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "goto-run":
            self.app.push_screen(RunScreen())
        elif event.button.id == "goto-gate":
            self.app.push_screen(GateScreen())
        elif event.button.id == "goto-reports":
            self.app.push_screen(ReportsScreen())
        elif event.button.id == "run-doctor":
            self.run_doctor()

    def run_doctor(self) -> None:
        log = self.query_one("#doctor-log", RichLog)
        log.clear()
        argv = cbench_command("doctor", [])
        log.write(f"[dim]$ {' '.join(argv)}[/dim]")
        self._run_doctor_worker(argv, log)

    @work(exclusive=True)
    async def _run_doctor_worker(self, argv, log: RichLog) -> None:
        await run_job(argv, on_line=lambda line: log.write(line))


class RunScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        with Vertical(id="run-form"):
            yield Static("Run a suite -- equivalent to running `cbench <suite>` yourself.")
            yield Select(
                [(label, subcmd) for label, subcmd in RUNNABLE_SUITES],
                id="suite-select", value=RUNNABLE_SUITES[0][1], allow_blank=False,
            )
            yield Input(placeholder="model tag, e.g. gemma3:12b", id="model-input")
            yield Checkbox("Dry run (print payload, call no model)", id="dry-run-checkbox", value=True)
            yield Input(
                placeholder="extra flags, e.g. --boundary both --sandbox extended",
                id="extra-args-input",
            )
            with Horizontal():
                yield Button("Run", id="run-button", variant="primary")
                yield Button("Back", id="back-button")
            yield Static("", id="command-preview")
            yield RichLog(id="run-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-button":
            self.app.pop_screen()
        elif event.button.id == "run-button":
            self._start_run()

    def _start_run(self) -> None:
        subcommand = self.query_one("#suite-select", Select).value
        model = self.query_one("#model-input", Input).value.strip()
        dry_run = self.query_one("#dry-run-checkbox", Checkbox).value
        extra = self.query_one("#extra-args-input", Input).value

        log = self.query_one("#run-log", RichLog)
        preview = self.query_one("#command-preview", Static)
        log.clear()

        if not model:
            log.write("[bold red]A model tag is required.[/bold red]")
            return

        try:
            args = build_args(model, dry_run, extra)
        except ValueError as e:
            log.write(f"[bold red]Could not parse extra flags: {e}[/bold red]")
            return

        argv = cbench_command(subcommand, args)
        preview.update(f"[dim]$ {' '.join(argv)}[/dim]")
        log.write(f"[dim]$ {' '.join(argv)}[/dim]")
        self._run_worker(argv, log)

    @work(exclusive=True)
    async def _run_worker(self, argv, log: RichLog) -> None:
        result = await run_job(argv, on_line=lambda line: log.write(line))
        if result.error:
            log.write(f"[bold red]{result.error}[/bold red]")
        else:
            style = "bold green" if result.returncode == 0 else "bold red"
            log.write(f"[{style}]exit code: {result.returncode}[/{style}]")


class GateScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        with Vertical(id="gate-form"):
            yield Static("Gate-check a model -- equivalent to running `cbench gate` yourself.")
            yield Input(placeholder="model tag, e.g. gemma3:12b", id="gate-model-input")
            yield Checkbox(
                "Save to local model catalogue overlay (models.json)",
                id="gate-save-checkbox", value=False,
            )
            with Horizontal():
                yield Button("Gate-check", id="gate-button", variant="primary")
                yield Button("Back", id="gate-back-button")
            yield RichLog(id="gate-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "gate-back-button":
            self.app.pop_screen()
        elif event.button.id == "gate-button":
            self._start_gate()

    def _start_gate(self) -> None:
        model = self.query_one("#gate-model-input", Input).value.strip()
        save = self.query_one("#gate-save-checkbox", Checkbox).value
        log = self.query_one("#gate-log", RichLog)
        log.clear()

        if not model:
            log.write("[bold red]A model tag is required.[/bold red]")
            return

        args = ["--model", model]
        if save:
            args.append("--save")
        argv = cbench_command("gate", args)
        log.write(f"[dim]$ {' '.join(argv)}[/dim]")
        self._run_worker(argv, log)

    @work(exclusive=True)
    async def _run_worker(self, argv, log: RichLog) -> None:
        result = await run_job(argv, on_line=lambda line: log.write(line))
        if result.error:
            log.write(f"[bold red]{result.error}[/bold red]")
        else:
            style = "bold green" if result.returncode == 0 else "bold red"
            log.write(f"[{style}]exit code: {result.returncode}[/{style}]")


class ReportsScreen(Screen):
    """Browses whatever `results/` already contains -- this screen reads
    files `cbench` wrote, it never generates or edits any of them."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        root = _results_root()
        with Horizontal(id="reports-body"):
            if root.exists():
                yield DirectoryTree(str(root), id="reports-tree")
            else:
                yield Static(
                    f"No results/ directory yet at {root} -- run a suite first.",
                    id="reports-empty",
                )
            with VerticalScroll(id="reports-viewer-container"):
                yield TextArea("", id="reports-viewer", read_only=True)
        yield Footer()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        path = Path(event.path)
        viewer = self.query_one("#reports-viewer", TextArea)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > 200_000:
                text = text[:200_000] + "\n... [truncated for display]"
            viewer.text = text
        except Exception as e:
            viewer.text = f"Could not read {path}: {e}"


class CBenchTUI(App):
    TITLE = "openllm-cbench"
    BINDINGS = [("q", "quit", "Quit")]

    CSS = """
    InvariantBar { background: $warning-darken-2; color: $text; padding: 0 1; }
    #dashboard-buttons { height: auto; }
    #dashboard-buttons Button { margin: 0 1 0 0; }
    #run-form, #gate-form { padding: 1; }
    RichLog { height: 1fr; border: solid $accent; }
    #reports-body { height: 1fr; }
    #reports-tree { width: 40%; }
    #reports-viewer-container { width: 60%; }
    """

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen())


def main() -> int:
    # No ImportError guard here: this module's own top-level `from textual
    # import ...` already requires textual to be installed just to finish
    # loading -- the friendly "optional dependency" message lives in
    # cli.py's _cmd_tui(), which imports this module inside a try/except
    # instead. See that function's docstring for why.
    CBenchTUI().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
