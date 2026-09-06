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
    Button, Checkbox, DataTable, DirectoryTree, Footer, Header, Input, RichLog,
    Select, Static, TextArea,
)

from openllm_cbench.core.invariant import SAFETY_INVARIANT
from openllm_cbench.tui.jobs import RUNNABLE_SUITES, build_args, cbench_command, run_job


def _results_root() -> Path:
    # Same precedence core.paths.results_dir() uses for a single suite,
    # applied to the shared results/ root the TUI's report browser lists.
    override = os.environ.get("OPENLLM_CBENCH_RESULTS_DIR")
    return Path(override) if override else Path.cwd() / "results"


async def _populate_model_select(select: Select) -> None:
    """Fills a model-picker Select with locally-pulled tags, via
    core.discover.list_local_models() -- the identical function `cbench
    discover` itself calls, not a reimplementation. Runs the blocking
    HTTP call in a thread (asyncio.to_thread) so it can't stall the UI;
    silent on failure (unreachable endpoint, etc) -- the picker just
    stays empty and the paired free-text Input still works regardless,
    same "degrade, don't block" discipline as everywhere else in this
    app."""
    import asyncio

    from openllm_cbench.core.discover import list_local_models

    try:
        local = await asyncio.to_thread(list_local_models)
    except Exception:
        select.set_options([])
        select.prompt = "Could not list local models — type the tag below"
        return
    options = [(m["name"], m["name"]) for m in sorted(local, key=lambda x: x["name"])]
    select.set_options(options)
    select.prompt = "Pick a local model (or type the tag below)" if options else \
        "No local models found — type the tag below"


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
                yield Button("Full assessment", id="goto-assess", variant="primary")
                yield Button("Gate a model", id="goto-gate")
                yield Button("Local models", id="goto-models")
            with Horizontal(id="dashboard-buttons-2"):
                yield Button("Browse reports", id="goto-reports")
                yield Button("Check environment", id="run-doctor")
                yield Button("About / extend this", id="goto-about")
            yield Static(
                "\"Check environment\" runs `cbench doctor`: confirms your endpoint is "
                "reachable, the safety canary binds correctly, reports GPU/RAM headroom, "
                "and how many models are in your catalogue. Calls no model. Safe to run "
                "any time; nothing else on this screen runs automatically.",
                id="doctor-caption",
            )
            yield RichLog(id="doctor-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        # Deliberately does NOT run `cbench doctor` automatically -- every
        # cbench action this TUI takes happens because of a click, none on
        # app startup, so what ran and why is never ambiguous.
        self.query_one("#doctor-log", RichLog).write(
            "[dim]Press \"Check environment\" above to run it.[/dim]"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "goto-run":
            self.app.push_screen(RunScreen())
        elif event.button.id == "goto-assess":
            self.app.push_screen(AssessmentScreen())
        elif event.button.id == "goto-gate":
            self.app.push_screen(GateScreen())
        elif event.button.id == "goto-models":
            self.app.push_screen(ModelsScreen())
        elif event.button.id == "goto-reports":
            self.app.push_screen(ReportsScreen())
        elif event.button.id == "run-doctor":
            self.run_doctor()
        elif event.button.id == "goto-about":
            self.app.push_screen(AboutScreen())

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
            yield Select([], id="model-select", allow_blank=True,
                         prompt="Pick a local model (or type the tag below) — loading...")
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

    def on_mount(self) -> None:
        self._load_models()

    @work(exclusive=True)
    async def _load_models(self) -> None:
        await _populate_model_select(self.query_one("#model-select", Select))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model-select" and event.value is not Select.BLANK:
            self.query_one("#model-input", Input).value = str(event.value)

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
            yield Select([], id="gate-model-select", allow_blank=True,
                         prompt="Pick a local model (or type the tag below) — loading...")
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

    def on_mount(self) -> None:
        self._load_models()

    @work(exclusive=True)
    async def _load_models(self) -> None:
        await _populate_model_select(self.query_one("#gate-model-select", Select))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "gate-model-select" and event.value is not Select.BLANK:
            self.query_one("#gate-model-input", Input).value = str(event.value)

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


class ModelsScreen(Screen):
    """What's actually available to run a suite against: every model
    pulled into the local endpoint, and whether the catalogue already
    knows about it. Reads the endpoint directly via core.discover (same
    function `cbench discover` calls) -- listing is read-only, no suite
    logic here. The one action this screen can take (gate + save the
    selected row) is a real `cbench gate --save` subprocess, same as
    every other action anywhere else in this app."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        with Vertical(id="models-body"):
            yield Static(
                "Models pulled into your local endpoint. \"Catalogued\" means the "
                "model catalogue already has config guidance for this exact tag -- "
                "an uncatalogued model still runs fine, just \"ungated\"."
            )
            with Horizontal(id="models-buttons"):
                yield Button("Refresh", id="models-refresh", variant="primary")
                yield Button("Gate + save selected", id="models-gate-selected")
                yield Button("Search / pull a new model", id="goto-pull")
                yield Button("Back", id="models-back")
            yield DataTable(id="models-table")
            yield RichLog(id="models-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#models-table", DataTable)
        table.add_columns("Tag", "Params (B)", "Quant", "Size", "Catalogued")
        table.cursor_type = "row"
        self._refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "models-back":
            self.app.pop_screen()
        elif event.button.id == "models-refresh":
            self._refresh()
        elif event.button.id == "goto-pull":
            self.app.push_screen(PullScreen())
        elif event.button.id == "models-gate-selected":
            self._gate_selected()

    def _refresh(self) -> None:
        self.query_one("#models-log", RichLog).write("[dim]Refreshing from the local endpoint...[/dim]")
        self._refresh_worker()

    @work(exclusive=True)
    async def _refresh_worker(self) -> None:
        import asyncio

        from openllm_cbench.core.discover import list_local_models, format_size
        from openllm_cbench.core.registry import load_registry

        table = self.query_one("#models-table", DataTable)
        log = self.query_one("#models-log", RichLog)
        try:
            local = await asyncio.to_thread(list_local_models)
        except Exception as e:
            log.write(f"[bold red]Could not list local models: {e}[/bold red]")
            return
        registry = await asyncio.to_thread(load_registry)
        catalogued = set(registry.get("models", {}).keys())

        table.clear()
        uncatalogued_n = 0
        for m in sorted(local, key=lambda x: x["name"]):
            is_cat = m["name"] in catalogued
            if not is_cat:
                uncatalogued_n += 1
            table.add_row(
                m["name"], str(m["params_b"]), m["quant"], format_size(m["size"]),
                "yes" if is_cat else "[bold yellow]no[/bold yellow]",
            )
        log.write(f"[dim]{len(local)} local model(s); {uncatalogued_n} not yet catalogued. "
                  f"Select a row and press \"Gate + save selected\" to catalogue one.[/dim]")

    def _gate_selected(self) -> None:
        table = self.query_one("#models-table", DataTable)
        log = self.query_one("#models-log", RichLog)
        if table.row_count == 0 or table.cursor_row is None:
            log.write("[bold red]No row selected -- click a model first.[/bold red]")
            return
        row = table.get_row_at(table.cursor_row)
        tag = str(row[0])
        argv = cbench_command("gate", ["--model", tag, "--save"])
        log.write(f"[dim]$ {' '.join(argv)}[/dim]")
        self._gate_worker(argv, log)

    @work(exclusive=True)
    async def _gate_worker(self, argv, log: RichLog) -> None:
        result = await run_job(argv, on_line=lambda line: log.write(line))
        if result.error:
            log.write(f"[bold red]{result.error}[/bold red]")
            return
        style = "bold green" if result.returncode == 0 else "bold red"
        log.write(f"[{style}]exit code: {result.returncode}[/{style}]")
        if result.returncode == 0:
            self._refresh()


class PullScreen(Screen):
    """Search Ollama's registry for a model, then optionally download it
    -- both are real `cbench search`/`cbench pull` subprocesses, same
    pattern as every other action-taking screen. Pull is the one action
    in this app that causes real, potentially large (multi-GB) network
    egress; the invariant bar and the static warning below both say so
    before that button is anywhere near a click. Search causes none --
    it aborts the same /api/pull request right after the manifest step,
    before any layer data downloads (see core/pull.py:
    check_model_availability())."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        with Vertical(id="pull-form"):
            yield Static(
                "Search Ollama's registry for an exact model tag (no download), then "
                "pull it if you want it -- equivalent to running `cbench search` / "
                "`cbench pull` (or `ollama pull`) yourself. [bold]Pull downloads real "
                "data from Ollama's registry[/bold], possibly several GB, and can take "
                "a while -- Search never does."
            )
            yield Input(placeholder="model tag, e.g. qwen3:4b", id="pull-model-input")
            with Horizontal():
                yield Button("Search", id="search-button", variant="primary")
                yield Button("Pull", id="pull-button")
                yield Button("Back", id="pull-back-button")
            yield RichLog(id="pull-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pull-back-button":
            self.app.pop_screen()
        elif event.button.id == "search-button":
            self._start_search()
        elif event.button.id == "pull-button":
            self._start_pull()

    def _start_search(self) -> None:
        model = self.query_one("#pull-model-input", Input).value.strip()
        log = self.query_one("#pull-log", RichLog)
        log.clear()
        if not model:
            log.write("[bold red]A model tag is required.[/bold red]")
            return
        argv = cbench_command("search", ["--model", model])
        log.write(f"[dim]$ {' '.join(argv)}[/dim]")
        self._search_worker(argv, log)

    @work(exclusive=True)
    async def _search_worker(self, argv, log: RichLog) -> None:
        result = await run_job(argv, on_line=lambda line: log.write(line))
        if result.error:
            log.write(f"[bold red]{result.error}[/bold red]")
        else:
            style = "bold green" if result.returncode == 0 else "bold red"
            log.write(f"[{style}]exit code: {result.returncode}[/{style}]")

    def _start_pull(self) -> None:
        model = self.query_one("#pull-model-input", Input).value.strip()
        log = self.query_one("#pull-log", RichLog)
        log.clear()
        if not model:
            log.write("[bold red]A model tag is required.[/bold red]")
            return
        argv = cbench_command("pull", ["--model", model])
        log.write(f"[dim]$ {' '.join(argv)}[/dim]")
        self._pull_worker(argv, log)

    @work(exclusive=True)
    async def _pull_worker(self, argv, log: RichLog) -> None:
        result = await run_job(argv, on_line=lambda line: log.write(line))
        if result.error:
            log.write(f"[bold red]{result.error}[/bold red]")
        else:
            style = "bold green" if result.returncode == 0 else "bold red"
            log.write(f"[{style}]exit code: {result.returncode}[/{style}]")
            if result.returncode == 0:
                log.write("[dim]Done. Go to \"Local models\" and gate + save it to add it "
                          "to your catalogue.[/dim]")


class AssessmentScreen(Screen):
    """Full assessment of one model -- equivalent to running `cbench
    assess` yourself, and literally is: a real subprocess, same as every
    other action-taking screen. All orchestration (repeated trials,
    auto-aggregation, report paths) lives in cli.py's _cmd_assess(), not
    here -- this screen is a form and a log, nothing more."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        with Vertical(id="assess-form"):
            yield Static(
                "Full assessment -- equivalent to running `cbench assess` yourself. Runs "
                "N trials of each selected suite, then auto-aggregates each into a "
                "trial-summary report and points you at it. S1+S2+S3 only -- this "
                "framework's own three suites (run `cbench assess --help` for why S4/S5 "
                "aren't included). Can take a while for a larger model or a high trial "
                "count -- start with Dry run checked to preview what will run."
            )
            yield Select([], id="assess-model-select", allow_blank=True,
                         prompt="Pick a local model (or type the tag below) — loading...")
            yield Input(placeholder="model tag, e.g. gemma3:12b", id="assess-model-input")
            with Horizontal(id="assess-suite-checks"):
                yield Checkbox("S1 containment", id="assess-s1", value=True)
                yield Checkbox("S2 channel", id="assess-s2", value=True)
                yield Checkbox("S3 persistence", id="assess-s3", value=True)
            yield Input(value="3", placeholder="trials per suite (default 3)", id="assess-trials-input")
            yield Checkbox("Dry run (print payloads, call no model)", id="assess-dry-run", value=True)
            with Horizontal():
                yield Button("Start assessment", id="assess-start", variant="primary")
                yield Button("Back", id="assess-back")
            yield Static("", id="assess-preview")
            yield RichLog(id="assess-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._load_models()

    @work(exclusive=True)
    async def _load_models(self) -> None:
        await _populate_model_select(self.query_one("#assess-model-select", Select))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "assess-model-select" and event.value is not Select.BLANK:
            self.query_one("#assess-model-input", Input).value = str(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "assess-back":
            self.app.pop_screen()
        elif event.button.id == "assess-start":
            self._start_assessment()

    def _start_assessment(self) -> None:
        model = self.query_one("#assess-model-input", Input).value.strip()
        log = self.query_one("#assess-log", RichLog)
        preview = self.query_one("#assess-preview", Static)
        log.clear()

        if not model:
            log.write("[bold red]A model tag is required.[/bold red]")
            return

        suites = []
        if self.query_one("#assess-s1", Checkbox).value:
            suites.append("s1")
        if self.query_one("#assess-s2", Checkbox).value:
            suites.append("s2")
        if self.query_one("#assess-s3", Checkbox).value:
            suites.append("s3")
        if not suites:
            log.write("[bold red]Select at least one suite.[/bold red]")
            return

        trials_raw = self.query_one("#assess-trials-input", Input).value.strip() or "3"
        try:
            trials = int(trials_raw)
            if trials < 1:
                raise ValueError
        except ValueError:
            log.write("[bold red]Trials must be a positive whole number.[/bold red]")
            return

        args = ["--model", model, "--suites", ",".join(suites), "--trials", str(trials)]
        if self.query_one("#assess-dry-run", Checkbox).value:
            args.append("--dry-run")

        argv = cbench_command("assess", args)
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
            if result.returncode == 0:
                log.write("[dim]Go to \"Browse reports\" to read the trial-summary reports "
                          "linked above.[/dim]")


_ABOUT_TEXT = """\
[bold]About this framework[/bold]

openllm-cbench was built collaboratively with an AI coding assistant \
(Claude Code) -- not as a demo of that, but because the discipline an \
assistant like that is good at (reading its own prior output critically, \
writing a regression test for every real bug before moving on, checking \
a claim against the actual code instead of memory) turned out to matter \
a lot for a tool whose whole job is measuring whether a model's stated \
capabilities match what it actually does.

[bold]Extending it yourself[/bold]

This project deliberately doesn't assume you use any one AI tool -- \
that's why it has a CONTRIBUTING.md instead of a tool-specific config \
file. If you want to extend a suite, add a model to the catalogue, or \
build a new scoring metric with an AI coding assistant's help, any of \
these work the same way: open this repo in [bold]Claude Code[/bold] or \
[bold]Claude Cowork[/bold], or in [bold]Codex CLI[/bold] / \
[bold]ChatGPT Cowork[/bold], and point it at:

  - ARCHITECTURE.md   -- what every suite measures and why, the control
                         inventory, and "how not to fool yourself with
                         this tool" (the most transferable section)
  - CONTRIBUTING.md    -- layout, conventions, and the standing rules
                         this project has learned the hard way (gate-check
                         both think states, never branch suite code on a
                         model name, smoke-test before calling something
                         done)
  - tests/             -- the parity and safety-invariant tests any
                         change should keep passing

None of this requires a specific vendor. The instructions above are \
written to make sense to a human reading them cold, and that's also \
what makes them make sense to any AI assistant you point at them.
"""


class AboutScreen(Screen):
    """Documentation, not a feature -- reads a static string, takes no
    action, contacts nothing. Exists because a request to 'encourage
    using an AI coding assistant to extend this' is better served by
    pointing at the docs that already answer it than by adding a new
    runtime dependency on any one vendor's CLI."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        with VerticalScroll(id="about-body"):
            yield Static(_ABOUT_TEXT)
        yield Button("Back", id="about-back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "about-back":
            self.app.pop_screen()


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
    #dashboard-buttons, #dashboard-buttons-2 { height: auto; }
    #dashboard-buttons Button, #dashboard-buttons-2 Button { margin: 0 1 1 0; }
    #doctor-caption { color: $text-muted; padding: 0 0 1 0; }
    #run-form, #gate-form, #models-body, #pull-form, #assess-form { padding: 1; }
    RichLog { height: 1fr; border: solid $accent; }
    #reports-body { height: 1fr; }
    #reports-tree { width: 40%; }
    #reports-viewer-container { width: 60%; }
    #models-buttons { height: auto; }
    #models-buttons Button { margin: 0 1 0 0; }
    #models-table { height: 12; border: solid $accent; }
    #assess-suite-checks { height: auto; }
    #assess-suite-checks Checkbox { margin: 0 2 0 0; }
    #about-body { height: 1fr; padding: 1 2; }
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
