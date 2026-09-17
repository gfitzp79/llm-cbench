"""
`cbench tui` -- a thin Textual control panel over the finished CLI.

Every screen here does one of two things: (1) shell out to a real
`cbench` subcommand via jobs.run_job() and stream its output, or (2)
read files `cbench` itself already wrote (results/, the model catalogue).
There is no suite logic in this file -- see jobs.py's module docstring
for why that split is load-bearing, not just tidy.
"""

import os
import re
import sys
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button, Checkbox, DataTable, DirectoryTree, Footer, Header, Input, ProgressBar, RichLog,
    Select, Static, TextArea,
)

from openllm_cbench.core.invariant import SAFETY_INVARIANT
from openllm_cbench.tui.jobs import (
    RUNNABLE_SUITES, build_args, cbench_command, condensed_line_filter, parse_trial_header,
    run_job, save_job_log,
)


def _results_root() -> Path:
    # Same precedence core.paths.results_dir() uses for a single suite,
    # applied to the shared results/ root the TUI's report browser lists.
    override = os.environ.get("OPENLLM_CBENCH_RESULTS_DIR")
    return Path(override) if override else Path.cwd() / "results"


async def _populate_model_select(select: Select) -> dict:
    """Fills a model-picker Select with locally-pulled tags, via
    core.discover.list_local_models() -- the identical function `cbench
    discover` itself calls, not a reimplementation. Runs the blocking
    HTTP call in a thread (asyncio.to_thread) so it can't stall the UI;
    silent on failure (unreachable endpoint, etc) -- the picker just
    stays empty and the paired free-text Input still works regardless,
    same "degrade, don't block" discipline as everywhere else in this
    app.

    Returns {tag -> the endpoint's own record} so a caller can reuse what
    was already fetched. That record carries params_b/quant/size for
    models the CATALOGUE has never seen, which is the only way the
    hardware fit warning can fire for a model someone just pulled -- the
    exact case it matters most."""
    import asyncio

    from openllm_cbench.core.discover import list_local_models

    from textual.css.query import NoMatches

    def _fill(options, prompt):
        """Mutating a Select after an await can outlive the screen.

        set_options() queries the Select's OWN children (SelectOverlay),
        so checking that the Select itself still resolves proves nothing
        -- its children are already gone when a screen is torn down
        mid-fetch, and the NoMatches raised here becomes a WorkerFailed
        that takes the app down. Seen as an intermittent failure in two
        unrelated-looking tests before it was traced. Every screen with a
        model picker funnels through this one function, so the guard
        belongs here rather than at each call site."""
        try:
            select.set_options(options)
            select.prompt = prompt
        except NoMatches:
            pass

    try:
        local = await asyncio.to_thread(list_local_models)
    except Exception:
        _fill([], "Could not list local models — type the tag below")
        return {}
    options = [(m["name"], m["name"]) for m in sorted(local, key=lambda x: x["name"])]
    _fill(options, "Pick a local model (or type the tag below)" if options else
          "No local models found — type the tag below")
    return {m["name"]: m for m in local}


def _report_job_result(log: RichLog, result, success_note: str = "") -> bool:
    """The tail every action-taking screen shares once its subprocess
    exits: save the full output to a real file and say where, then report
    either the failure-to-start error or a colour-coded exit code, plus an
    optional "here's where to look next" line on success only. Returns
    True if the job actually succeeded, so a caller can chain off that
    (ModelsScreen refreshes its table on a successful gate).

    Was copy-pasted verbatim across six screens before this existed --
    identical enough that a change to how a job reports itself had to be
    made in six places or become inconsistent in five."""
    log.write(f"[dim]Full log saved to {save_job_log(result)}[/dim]")
    if result.error:
        log.write(f"[bold red]{result.error}[/bold red]")
        return False
    style = "bold green" if result.returncode == 0 else "bold red"
    log.write(f"[{style}]exit code: {result.returncode}[/{style}]")
    if result.returncode == 0 and success_note:
        log.write(f"[dim]{success_note}[/dim]")
    return result.returncode == 0


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
                yield Button("Score a model", id="goto-score", variant="primary")
                yield Button("Gate a model", id="goto-gate")
                yield Button("Local models", id="goto-models")
            with Horizontal(id="dashboard-buttons-2"):
                yield Button("Browse reports", id="goto-reports")
                yield Button("Share / validate results", id="goto-community")
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
        elif event.button.id == "goto-score":
            self.app.push_screen(ScoreScreen())
        elif event.button.id == "goto-gate":
            self.app.push_screen(GateScreen())
        elif event.button.id == "goto-models":
            self.app.push_screen(ModelsScreen())
        elif event.button.id == "goto-reports":
            self.app.push_screen(ReportsScreen())
        elif event.button.id == "goto-community":
            self.app.push_screen(CommunityScreen())
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
        result = await run_job(argv, on_line=lambda line: log.write(line))
        log.write(f"[dim]Full log saved to {save_job_log(result)}[/dim]")


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
        _report_job_result(log, result)


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
        _report_job_result(log, result)


class ModelsScreen(Screen):
    """What's actually available to run a suite against: every model
    pulled into the local endpoint, and whether the catalogue already
    knows about it. Reads the endpoint directly via core.discover (same
    function `cbench discover` calls) -- listing is read-only, no suite
    logic here. The two actions this screen can take are both real
    subprocesses, same as every other action anywhere else in this app:
    gate + save the selected row (`cbench gate --save`), and gate + save
    every uncatalogued row at once (`cbench discover --gate-all`)."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        with Vertical(id="models-body"):
            yield Static(
                "Models pulled into your local endpoint. \"Catalogued\" means the "
                "model catalogue already has config guidance for this exact tag -- "
                "an uncatalogued model still runs fine, just \"ungated\".\n"
                "Speed column: a bold tok/s figure is MEASURED on this machine during that "
                "model's gate check. A word (fast/good/moderate/slow) is an estimate from "
                "the parameters actually read per token -- for a mixture-of-experts model "
                "that's the active experts, not the full weight count, which is why a "
                "30B-A3B MoE outruns a dense 30B of the same size on disk.\n"
                "Score column: an A-F grade (0-100), the worst of the three suites "
                "run -- not an average, see \"Scoring a model\" in README.md. "
                "\"not scored\" = never run through `cbench score`. \"N/A\" = nothing "
                "gradable yet. \"INVALID\" = a validity guard fired (e.g. mismatched "
                "task sets) -- that suite is excluded from the grade, don't trust it "
                "yet regardless. Trailing \"*\" = an otherwise-ok suite still has an "
                "unresolved caveat -- read the full scorecard "
                "(results/scorecards/<tag>.md) before citing the grade alone.",
                id="models-score-legend",
            )
            with Horizontal(id="models-buttons"):
                yield Button("Refresh", id="models-refresh", variant="primary")
                yield Button("Gate + save selected", id="models-gate-selected")
                yield Button("Gate + save all uncatalogued", id="models-gate-all")
                yield Button("Search / pull a new model", id="goto-pull")
                yield Button("Back", id="models-back")
            yield DataTable(id="models-table")
            yield RichLog(id="models-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#models-table", DataTable)
        table.add_columns("Tag", "Params (B)", "Quant", "Size", "Fit", "Speed", "Catalogued", "Score")
        table.cursor_type = "row"
        self._hw_info = None
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
        elif event.button.id == "models-gate-all":
            self._gate_all()

    def _refresh(self) -> None:
        # Guarded for the same reason as _populate_model_select and
        # _refresh_catalogue_status: _gate_worker calls this after awaiting
        # a real `cbench gate` / `cbench discover --gate-all`, which is a
        # live model call and so the widest await window in the app. If the
        # screen closed during it, this query raises inside a worker and
        # Textual turns that into WorkerFailed.
        from textual.css.query import NoMatches
        try:
            log = self.query_one("#models-log", RichLog)
        except NoMatches:
            return
        log.write("[dim]Refreshing from the local endpoint...[/dim]")
        self._refresh_worker()

    @work(exclusive=True)
    async def _refresh_worker(self) -> None:
        import asyncio

        from openllm_cbench.core.discover import list_local_models, format_size
        from openllm_cbench.core.registry import load_registry
        from openllm_cbench.scoring.scorecard import catalogue_compact_label

        table = self.query_one("#models-table", DataTable)
        log = self.query_one("#models-log", RichLog)
        try:
            local = await asyncio.to_thread(list_local_models)
        except Exception as e:
            log.write(f"[bold red]Could not list local models: {e}[/bold red]")
            return
        registry = await asyncio.to_thread(load_registry)
        catalogued = set(registry.get("models", {}).keys())

        from openllm_cbench.core.hardware import fit_assessment, performance_estimate, probe
        if self._hw_info is None:
            self._hw_info = await asyncio.to_thread(probe)
        vram_mb = (self._hw_info or {}).get("gpu_vram_mb")

        table.clear()
        uncatalogued_n = 0
        spills_n = 0
        for m in sorted(local, key=lambda x: x["name"]):
            is_cat = m["name"] in catalogued
            if not is_cat:
                uncatalogued_n += 1
            score = await asyncio.to_thread(catalogue_compact_label, m["name"])
            assessment = fit_assessment(
                vram_mb, tag=m["name"], architecture=m.get("architecture"),
                params_b=m.get("params_b"), quant=m.get("quant"),
                size_mb=round(m["size"] / (1024 * 1024)) if m.get("size") else None,
            )
            if assessment["tier"] == "spills":
                spills_n += 1
            fit_cell = {
                "fits": "[green]fits[/green]",
                "tight": "[yellow]tight[/yellow]",
                "spills": f"[bold red]{assessment['headline']}[/bold red]",
                "unknown": "[dim]?[/dim]",
            }[assessment["tier"]]

            # Measured beats estimated: a gate check on this machine
            # timed the model, so use that rather than inferring speed
            # from parameter count.
            entry = registry.get("models", {}).get(m["name"]) or {}
            perf = performance_estimate(
                fit_tier=assessment["tier"],
                params_b=m.get("params_b"),
                active_params_b=assessment.get("active_params_b"),
                measured_tok_s=entry.get("measured_tok_s"),
                moe=assessment.get("moe", False),
            )
            if perf["source"] == "measured":
                speed_cell = f"[bold green]{perf['tok_s']:.0f} tok/s[/bold green]"
            else:
                speed_cell = {
                    "fast": "[green]fast[/green]",
                    "good": "[green]good[/green]",
                    "moderate": "[yellow]moderate[/yellow]",
                    "slow": "[red]slow[/red]",
                    "unknown": "[dim]?[/dim]",
                }[perf["label"]]

            table.add_row(
                m["name"], str(m["params_b"]), m["quant"], format_size(m["size"]), fit_cell,
                speed_cell,
                "yes" if is_cat else "[bold yellow]no[/bold yellow]",
                score if score != "not scored" else "[dim]not scored[/dim]",
            )
        if spills_n:
            log.write(f"[bold yellow]{spills_n} model(s) won't fit in this GPU's "
                       f"{vram_mb:,} MB of VRAM and will run partly on CPU -- much slower, but "
                       f"still valid. \"spills (MoE)\" degrades far less than a dense model of "
                       f"the same size, since only a fraction of its parameters are active per "
                       f"token.[/bold yellow]")
        log.write(f"[dim]{len(local)} local model(s); {uncatalogued_n} not yet catalogued. "
                  f"Select a row and press \"Gate + save selected\" to catalogue one, or "
                  f"\"Gate + save all uncatalogued\" to do all {uncatalogued_n} in one batch "
                  f"(real model calls, one at a time). \"Score a model\" from the dashboard "
                  f"fills in the Score column above (legend above the table explains what it "
                  f"means).[/dim]")

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

    def _gate_all(self) -> None:
        """`cbench discover --gate-all` -- the batch equivalent of pressing
        "Gate + save selected" once per uncatalogued row. Deliberately the
        real subcommand rather than a TUI-side loop over _gate_selected():
        the ordering, the --limit bound, and the per-model failure handling
        already live in cli.py's _cmd_discover(), and a second
        implementation of them here is exactly what this app's own design
        invariant exists to prevent."""
        log = self.query_one("#models-log", RichLog)
        argv = cbench_command("discover", ["--gate-all"])
        log.write(f"[dim]$ {' '.join(argv)}[/dim]")
        log.write("[dim]Gate-checking every uncatalogued model, one at a time -- real model "
                  "calls, so this can take a while for a long list.[/dim]")
        self._gate_worker(argv, log)

    @work(exclusive=True)
    async def _gate_worker(self, argv, log: RichLog) -> None:
        result = await run_job(argv, on_line=lambda line: log.write(line))
        if _report_job_result(log, result):
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
        _report_job_result(log, result)

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
        _report_job_result(log, result,
                            "Done. Go to \"Local models\" and gate + save it to add it "
                            "to your catalogue.")


class ScoreScreen(Screen):
    """`cbench score` -- a real subprocess, same as every other
    action-taking screen. All the actual depth->trials mapping, the
    scorecard computation, and the run-lock live in cli.py's _cmd_score()
    and scoring/scorecard.py, not here; this screen is a form and a log.
    The resulting scorecard is what the Models browser's Score column
    (ModelsScreen) reads afterward -- run this first, then check there or
    in "Browse reports"."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        with Vertical(id="score-form"):
            yield Static(
                "Score a model -- equivalent to running `cbench score` yourself. Runs "
                "S1/S2/S3 at the chosen depth (or reads existing CSVs with \"From "
                "existing\", making no model call), then saves an A-F grade (worst of the "
                "three suites, not an average) plus the full per-suite detail underneath "
                "it -- see README.md \"Scoring a model\". The Models browser's Score column "
                "and `cbench catalogue` both read whatever this produces.\n"
                "Depth covers 1/3/5 trials. For any other trial count, run `cbench assess "
                "--model <tag> --trials N` in a terminal -- that runs the identical suites "
                "and aggregation this does, just without producing a grade afterward."
            )
            yield Select([], id="score-model-select", allow_blank=True,
                         prompt="Pick a local model (or type the tag below) — loading...")
            yield Input(placeholder="model tag, e.g. gemma3:12b", id="score-model-input")
            yield Static("", id="score-catalogue-status")
            yield Static("", id="score-hardware-status")
            with Horizontal(id="score-suite-checks"):
                yield Checkbox("S1 containment", id="score-s1", value=True)
                yield Checkbox("S2 channel", id="score-s2", value=True)
                yield Checkbox("S3 persistence", id="score-s3", value=True)
            yield Select(
                [("quick — 1 trial (exploratory only, below this framework's own "
                  "3-trial citability minimum)", "quick"),
                 ("standard — 3 trials (default; this framework's own pre-registered "
                  "minimum for a rate worth citing)", "standard"),
                 ("thorough — 5 trials (matches the extension-rule's own EXTEND target)",
                  "thorough")],
                id="score-depth-select", value="standard", allow_blank=False,
            )
            yield Checkbox(
                "From existing (score whatever CSVs are already on disk for this tag -- "
                "runs nothing, calls no model; depth above is ignored)",
                id="score-from-existing", value=False,
            )
            yield Checkbox("Dry run (print payloads, call no model)", id="score-dry-run", value=True)
            yield Checkbox(
                "Gate-check first if not catalogued (runs `cbench gate --save` before "
                "scoring, so a net-new model isn't silently UNGATED -- recommended, "
                "especially for a model this catalogue has never seen)",
                id="score-gate-first", value=True,
            )
            with Horizontal():
                yield Button("Score", id="score-start", variant="primary")
                yield Button("Back", id="score-back")
            yield Static("", id="score-preview")
            progress = ProgressBar(id="score-progress", show_eta=True)
            progress.display = False
            yield progress
            yield RichLog(id="score-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._hw_info = None
        self._local_models = {}
        self._load_models()
        self._probe_hardware()

    @work(exclusive=True)
    async def _load_models(self) -> None:
        from textual.css.query import NoMatches
        try:
            select = self.query_one("#score-model-select", Select)
        except NoMatches:
            return
        self._local_models = await _populate_model_select(select)
        # The endpoint's own record is what makes the fit warning work for
        # an uncatalogued model, so re-render the status line once it
        # lands -- the user may already have typed a tag by then.
        try:
            model = self.query_one("#score-model-input", Input).value.strip()
        except NoMatches:
            return
        if model:
            self._refresh_catalogue_status(model)

    @work(exclusive=True, group="hardware-probe")
    async def _probe_hardware(self) -> None:
        # A real subprocess call (nvidia-smi/rocm-smi) -- offloaded via
        # to_thread so it can't stall the UI, and done once per screen
        # visit rather than on every keystroke in the model-tag input.
        # Advisory only, same as core/hardware.py's own module docstring:
        # this never blocks or gates a run, it only informs the fit
        # warning shown alongside the catalogue status.
        import asyncio as _asyncio

        from openllm_cbench.core.hardware import probe
        self._hw_info = await _asyncio.to_thread(probe)
        # The screen can be gone by the time the probe returns -- a user
        # who opens Score and presses Escape immediately beats an
        # nvidia-smi call comfortably, and app teardown does too.
        # Touching the DOM after that raises NoMatches inside a worker,
        # which Textual escalates into WorkerFailed and takes the whole
        # app down with it. Found exactly that way: an intermittent test
        # failure that looked like a Windows teardown flake for two runs
        # before it reproduced deliberately.
        #
        # Catching NoMatches rather than checking is_mounted first: during
        # teardown the screen still reports itself mounted while its
        # children are already gone, so the flag says yes and the query
        # still raises. The exception is the only honest signal.
        from textual.css.query import NoMatches
        try:
            model = self.query_one("#score-model-input", Input).value.strip()
        except NoMatches:
            return
        if model:
            self._refresh_catalogue_status(model)

    def _hardware_fit_line(self, model, entry) -> str:
        """Renders a hardware fit warning against this machine's detected
        VRAM -- blank only when neither side of the comparison is
        knowable. Never a reason to refuse a run, only to set
        expectations: a model that spills into system RAM turns a
        20-minute run into an overnight one, and makes the progress bar's
        ETA meaningless.

        Reads the CATALOGUE first, then falls back to the endpoint's own
        record of locally-pulled models. That fallback is the whole point:
        the warning used to read the catalogue alone, so it stayed silent
        for any model not catalogued yet -- i.e. the freshly-pulled 30B
        someone is about to discover the hard way is too big for their
        GPU. Found live on exactly that case."""
        if not self._hw_info:
            return ""
        from openllm_cbench.core.hardware import fit_assessment

        local = (self._local_models or {}).get(model, {})
        size_bytes = local.get("size")
        assessment = fit_assessment(
            self._hw_info.get("gpu_vram_mb"),
            tag=model,
            architecture=(entry or {}).get("architecture") or local.get("architecture"),
            params_b=(entry or {}).get("params_b") or local.get("params_b"),
            quant=(entry or {}).get("quant") or local.get("quant"),
            size_mb=round(size_bytes / (1024 * 1024)) if size_bytes else None,
        )
        if assessment["tier"] == "unknown":
            return ""
        if assessment["tier"] == "fits":
            return f"[dim]{assessment['note']}[/dim]"
        colour = "bold yellow" if assessment["tier"] == "spills" else "yellow"
        return f"[{colour}]⚠ {assessment['note']}[/{colour}]"

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "score-model-select" and event.value is not Select.BLANK:
            self.query_one("#score-model-input", Input).value = str(event.value)
            self._refresh_catalogue_status(str(event.value))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "score-model-input":
            self._refresh_catalogue_status(event.value.strip())

    def _refresh_catalogue_status(self, model: str) -> None:
        """Local-only registry read (same function `cbench discover`/
        `cbench catalogue` already call), synchronous -- reading two small
        JSON files off disk doesn't need a worker the way a live endpoint
        call does. Shows catalogue status BEFORE a run starts, not just in
        the log after one is already underway -- a model this catalogue
        has never seen is going to be the NORM as community-submitted
        results bring in models nobody here has gated yet, not an edge
        case worth discovering only mid-run.

        Tolerates the screen being gone. Two separate workers call this
        after an await (_probe_hardware after timing an nvidia-smi call,
        _load_models after listing the endpoint's models), so the screen
        can be torn down between the await returning and this running.
        The guard was originally written at both call sites and NOT here,
        which fixed neither: both funnel into this function, and its own
        query_one raised from inside the guarded caller. A duplicated
        predicate has to be fixed where it actually lives."""
        from textual.css.query import NoMatches
        try:
            status = self.query_one("#score-catalogue-status", Static)
            hardware = self.query_one("#score-hardware-status", Static)
        except NoMatches:
            return
        if not model:
            status.update("")
            hardware.update("")
            return
        from openllm_cbench.core.registry import load_registry, lookup
        entry = lookup(model, load_registry())
        if entry is not None:
            status.update("[green]✓ catalogued[/green] -- config guidance on file for this tag.")
        else:
            status.update(
                "[bold yellow]⚠ not in your model catalogue[/bold yellow] -- this run will be "
                "UNGATED unless \"Gate-check first\" below is checked (it is, by default)."
            )
        hardware.update(self._hardware_fit_line(model, entry))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "score-back":
            self.app.pop_screen()
        elif event.button.id == "score-start":
            self._start_score()

    def _start_score(self) -> None:
        model = self.query_one("#score-model-input", Input).value.strip()
        log = self.query_one("#score-log", RichLog)
        preview = self.query_one("#score-preview", Static)
        log.clear()

        if not model:
            log.write("[bold red]A model tag is required.[/bold red]")
            return

        suites = []
        if self.query_one("#score-s1", Checkbox).value:
            suites.append("s1")
        if self.query_one("#score-s2", Checkbox).value:
            suites.append("s2")
        if self.query_one("#score-s3", Checkbox).value:
            suites.append("s3")
        if not suites:
            log.write("[bold red]Select at least one suite.[/bold red]")
            return

        from_existing = self.query_one("#score-from-existing", Checkbox).value

        if from_existing:
            # "From existing" runs nothing and makes no model call by
            # design -- but that means it can also silently produce grade
            # N/A with exit code 0 if there was never anything on disk to
            # read, which looks identical to a successful run at a
            # glance. Found live: a brand-new model tag with "From
            # existing" left checked (e.g. carried over from a previous
            # run) scored N/A with no explanation why. Check what's
            # actually on disk BEFORE running, not after.
            # Read the CURRENT results dir (core.paths.results_dir() reads
            # $OPENLLM_CBENCH_RESULTS_DIR fresh on every call), not
            # aggregate.py's own S1_DIR/S2_DIR/S3_DIR -- those are frozen
            # at whatever the env var was when that module first got
            # imported (see _cmd_score's own docstring on why --from-existing
            # needs the env var set before the process starts), which is
            # the right call for a real `cbench score` subprocess but
            # would silently ignore this screen's own test isolation.
            from openllm_cbench.core.paths import results_dir
            from openllm_cbench.scoring.aggregate import find_csvs, model_tag
            suite_dirs = {
                "s1": (results_dir("s1_containment"), "containment"),
                "s2": (results_dir("s2_channel"), "channel"),
                "s3": (results_dir("s3_persistence"), "persistence"),
            }
            tag = model_tag(model)
            missing = [s for s in suites if not find_csvs(suite_dirs[s][0], suite_dirs[s][1], tag)]
            if len(missing) == len(suites):
                log.write(
                    f"[bold red]\"From existing\" is checked, but no "
                    f"{'/'.join(s.upper() for s in suites)} CSVs exist yet for '{model}' -- there is "
                    f"nothing on disk to score. This would run nothing and produce grade N/A. Uncheck "
                    f"\"From existing\" (and pick a depth) to actually run trials.[/bold red]"
                )
                return
            elif missing:
                log.write(
                    f"[bold yellow]\"From existing\": no {'/'.join(s.upper() for s in missing)} CSVs "
                    f"exist yet for '{model}' -- {'those' if len(missing) > 1 else 'that'} suite will "
                    f"show 'not run' below; the grade will only reflect whichever suite(s) do have "
                    f"data.[/bold yellow]"
                )

        args = ["--model", model, "--suites", ",".join(suites)]
        total_trials = None
        if from_existing:
            args.append("--from-existing")
        else:
            depth = self.query_one("#score-depth-select", Select).value
            args += ["--depth", str(depth)]
            if self.query_one("#score-dry-run", Checkbox).value:
                args.append("--dry-run")
            from openllm_cbench.scoring.scorecard import DEPTH_TRIALS
            total_trials = len(suites) * DEPTH_TRIALS[depth]

        argv = cbench_command("score", args)
        preview.update(f"[dim]$ {' '.join(argv)}[/dim]")
        log.write(f"[dim]$ {' '.join(argv)}[/dim]")

        progress = self.query_one("#score-progress", ProgressBar)
        if total_trials:
            # Total is known upfront (suites x trials-for-depth); progress
            # advances one unit per completed trial as `--- suite trial
            # N/M ---` headers stream past (see _run_worker). Textual's
            # own ProgressBar computes ETA from the observed rate of
            # .advance() calls, so it only starts reporting one once the
            # first trial has actually finished -- exactly the "don't
            # guess before you have a real data point" behaviour wanted
            # here, with no hand-rolled timing code needed.
            progress.update(total=total_trials, progress=0)
            progress.display = True
        else:
            progress.display = False

        # Gate first if this tag has never been seen and the checkbox says
        # to -- a convenience, not a requirement: this framework never
        # refuses to run against an unlisted or failed-gate model, it only
        # warns, so a gate failure below doesn't block the score run either.
        gate_first_argv = None
        if self.query_one("#score-gate-first", Checkbox).value:
            from openllm_cbench.core.registry import load_registry, lookup
            if lookup(model, load_registry()) is None:
                gate_first_argv = cbench_command("gate", ["--model", model, "--save"])

        self._run_worker(model, argv, log, progress, gate_first_argv)

    @work(exclusive=True)
    async def _run_worker(self, model, argv, log: RichLog, progress: ProgressBar,
                           gate_first_argv=None) -> None:
        if gate_first_argv:
            log.write("[dim]Model not catalogued -- gate-checking first "
                      "(uncheck \"Gate-check first\" to skip this):[/dim]")
            log.write(f"[dim]$ {' '.join(gate_first_argv)}[/dim]")
            gate_result = await run_job(gate_first_argv, on_line=lambda line: log.write(line))
            log.write(f"[dim]Full log saved to {save_job_log(gate_result)}[/dim]")

            from openllm_cbench.core.gate import summarize_gate_output
            from openllm_cbench.core.registry import load_registry, lookup
            summary = summarize_gate_output(gate_result.lines)
            if gate_result.error:
                log.write(f"[bold red]Gate-check process itself failed to run: {gate_result.error} -- "
                          f"nothing was learned about this model automatically.[/bold red]")
            elif summary["hard_failure"]:
                log.write(
                    "[bold red]Gate-check could not reach the endpoint for this model "
                    f"({summary['reason']}) -- the score run below makes the identical call and "
                    "will most likely fail the exact same way. Check the endpoint/model tag before "
                    "waiting on it.[/bold red]"
                )
            elif summary["clean"] is False and summary["caveats"]:
                log.write(
                    f"[bold yellow]Gate-check ran and found {len(summary['caveats'])} "
                    "caveat(s) -- proceeding to score anyway (this framework never blocks a run "
                    "on gate status), but read these first:[/bold yellow]"
                )
                for c in summary["caveats"]:
                    log.write(f"[bold yellow]  - {c}[/bold yellow]")
            elif summary["clean"]:
                log.write("[green]Gate-check clean -- no caveats.[/green]")
            else:
                log.write("[bold yellow]Gate-check exited non-zero but printed no recognizable "
                          "report -- proceeding to score anyway; see the full log above for "
                          "whatever it did print.[/bold yellow]")
            # The gate check just wrote (--save) a fresh catalogue entry
            # for this tag if it got far enough to -- re-read it so the
            # hardware fit line reflects real params_b/quant instead of
            # staying blank for the rest of this run.
            # Same teardown hazard as _probe_hardware, and the window is
            # much wider here: a gate check is a live endpoint call.
            from textual.css.query import NoMatches
            fresh_entry = lookup(model, load_registry())
            try:
                self.query_one("#score-hardware-status", Static).update(
                    self._hardware_fit_line(model, fresh_entry))
            except NoMatches:
                return
            log.write("")

        should_show = condensed_line_filter()
        trial_headers_seen = 0

        def on_line(line):
            nonlocal trial_headers_seen
            if parse_trial_header(line) is not None:
                if trial_headers_seen > 0:
                    progress.advance(1)
                trial_headers_seen += 1
            if should_show(line):
                log.write(line)

        result = await run_job(argv, on_line=on_line)
        if progress.display and progress.total:
            progress.update(progress=progress.total)
        _report_job_result(log, result,
                            "Check \"Local models\" for the updated Score column, or "
                            "\"Browse reports\" for the full scorecard.")


class CommunityScreen(Screen):
    """The three community-submission actions -- package, validate,
    submit -- each a real `cbench community-*` subprocess, same as every
    other action-taking screen. All the actual logic (what a submission
    is shaped like, what gets checksummed, how a PR is opened) lives in
    core/community.py and core/community_submit.py; this screen is a
    folder picker, a form, and a log.

    Submit deliberately mirrors the CLI's own two-step: pressing it
    previews the exact command sequence and sends nothing, and only the
    explicit "confirm" checkbox adds --confirm. This is the one action in
    the app that creates public content under the user's own name, so a
    single misplaced click must not be able to do it."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        with Vertical(id="community-form"):
            yield Static(
                "Share results for a model, or check someone else's submission.\n"
                "[bold]Package[/bold] (`cbench community-package`) bundles the CSVs already "
                "on disk for a model tag into a submittable folder, fills in submission.json, "
                "and records a SHA-256 per file. Uploads nothing.\n"
                "[bold]Validate[/bold] (`cbench community-validate`) checks a folder is shaped "
                "correctly. Read-only.\n"
                "[bold]Submit[/bold] (`cbench community-submit`) opens it as a pull request via "
                "your own authenticated `gh` -- previews only, until \"confirm\" below is "
                "checked.\n"
                "[bold]Raw data only:[/bold] a submission carries CSVs, never a grade. Anyone who "
                "wants a score runs `cbench score --from-existing` against the rows themselves. "
                "See community-results/README.md."
            )
            root = Path.cwd() / "community-results"
            with Horizontal(id="community-body"):
                if root.exists():
                    yield DirectoryTree(str(root), id="community-tree")
                else:
                    yield Static(
                        f"No community-results/ directory at {root}.\n\n"
                        f"This is created the first time you package a submission. If you "
                        f"have packaged one before, you are probably in a different working "
                        f"directory than when you did -- this path is resolved relative to "
                        f"where you launched cbench.",
                        id="community-tree-empty",
                    )
                with Vertical(id="community-form-inner"):
                    yield Select([], id="community-model-select", allow_blank=True,
                                  prompt="Model to package — scanning results...")
                    yield Select([], id="community-submission-select", allow_blank=True,
                                  prompt="Packaged submission to validate/submit — scanning...")
                    yield Input(
                        placeholder="(or type a model tag)",
                        id="community-model-input",
                    )
                    yield Input(
                        placeholder="(or type a submission folder path)",
                        id="community-path-input",
                    )
                    yield Checkbox(
                        "Accept contributor terms (right to share, no confidential data, "
                        "accurate hardware, Apache-2.0 licence grant, published permanently) "
                        "-- required before a package can be submitted",
                        id="community-terms", value=False,
                    )
                    yield Checkbox(
                        "Confirm submit -- actually fork, push and open a PUBLIC pull "
                        "request as you (unchecked = preview the commands only)",
                        id="community-confirm", value=False,
                    )
                    with Horizontal():
                        yield Button("Package", id="community-package", variant="primary")
                        yield Button("Validate", id="community-start", variant="primary")
                        yield Button("Submit", id="community-submit")
                        yield Button("Back", id="community-back")
                    yield Static("", id="community-preview")
                    yield RichLog(id="community-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_pickers()

    @work(exclusive=True)
    async def _refresh_pickers(self) -> None:
        """Fills both pickers from what actually exists: models that have
        CSVs on disk, and submissions already packaged. A user should be
        choosing from real options, not recalling a path."""
        import asyncio

        from textual.css.query import NoMatches

        from openllm_cbench.core.community import (
            list_packaged_submissions, models_with_local_results,
        )
        from openllm_cbench.core.discover import list_local_models

        def gather():
            try:
                tags = [m["name"] for m in list_local_models()]
            except Exception:
                tags = []
            return models_with_local_results(known_tags=tags), list_packaged_submissions()

        try:
            models, submissions = await asyncio.to_thread(gather)
        except Exception:
            models, submissions = [], []

        try:
            model_select = self.query_one("#community-model-select", Select)
            sub_select = self.query_one("#community-submission-select", Select)
        except NoMatches:
            return

        opts = []
        for tag, suites in models:
            n = sum(suites.values())
            opts.append((f"{tag}  ({n} CSV(s) across {len(suites)} suite(s))", tag))

        sub_opts = [
            (f"{d['model']}  {d['date']}  ({'terms accepted' if d['accepted'] else 'terms NOT accepted'})",
             str(d["path"]))
            for d in submissions
        ]

        # Same teardown hazard as _populate_model_select: set_options()
        # queries the Select's own children, which are gone if the screen
        # closed while the scan above was running.
        try:
            model_select.set_options(opts)
            model_select.prompt = ("Model to package" if opts else
                                    "No model has results on disk yet — score one first")
            sub_select.set_options(sub_opts)
            sub_select.prompt = ("Packaged submission to validate/submit" if sub_opts else
                                  "Nothing packaged yet — use Package first")
        except NoMatches:
            return

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value is Select.BLANK:
            return
        if event.select.id == "community-model-select":
            self.query_one("#community-model-input", Input).value = str(event.value)
        elif event.select.id == "community-submission-select":
            self.query_one("#community-path-input", Input).value = str(event.value)

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        if event.control.id == "community-tree":
            self.query_one("#community-path-input", Input).value = str(event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "community-back":
            self.app.pop_screen()
        elif event.button.id == "community-start":
            self._start_validate()
        elif event.button.id == "community-package":
            self._start_package()
        elif event.button.id == "community-submit":
            self._start_submit()

    def _launch(self, subcommand, args, log, preview):
        argv = cbench_command(subcommand, args)
        preview.update(f"[dim]$ {' '.join(argv)}[/dim]")
        log.write(f"[dim]$ {' '.join(argv)}[/dim]")
        self._run_worker(argv, log)

    def _start_validate(self) -> None:
        path = self.query_one("#community-path-input", Input).value.strip()
        log = self.query_one("#community-log", RichLog)
        preview = self.query_one("#community-preview", Static)
        log.clear()

        if not path:
            log.write("[bold red]A submission folder path is required -- click one in the "
                       "tree on the left, or type it.[/bold red]")
            return

        self._launch("community-validate", [path], log, preview)

    def _start_package(self) -> None:
        model = self.query_one("#community-model-input", Input).value.strip()
        log = self.query_one("#community-log", RichLog)
        preview = self.query_one("#community-preview", Static)
        log.clear()

        if not model:
            log.write("[bold red]A model tag is required to package -- type the tag whose "
                       "results you want to bundle.[/bold red]")
            return

        args = ["--model", model, "--zip"]
        if self.query_one("#community-terms", Checkbox).value:
            args.append("--accept-terms")
        else:
            log.write("[dim]Contributor terms not accepted -- the folder will be built so you "
                       "can read it, but it won't validate until you tick the terms box and "
                       "package again.[/dim]")
        self._launch("community-package", args, log, preview)

    def _start_submit(self) -> None:
        path = self.query_one("#community-path-input", Input).value.strip()
        confirm = self.query_one("#community-confirm", Checkbox).value
        log = self.query_one("#community-log", RichLog)
        preview = self.query_one("#community-preview", Static)
        log.clear()

        if not path:
            log.write("[bold red]A submission folder path is required -- package one first, "
                       "then click it in the tree on the left.[/bold red]")
            return

        args = [path]
        if confirm:
            args.append("--confirm")
            log.write("[bold yellow]\"Confirm submit\" is checked -- this will open a PUBLIC "
                       "pull request under your own GitHub account.[/bold yellow]")
        else:
            log.write("[dim]Preview only -- nothing will be sent. Check \"Confirm submit\" "
                       "above to actually open the pull request.[/dim]")
        self._launch("community-submit", args, log, preview)

    @work(exclusive=True)
    async def _run_worker(self, argv, log: RichLog) -> None:
        result = await run_job(argv, on_line=lambda line: log.write(line))
        ok = _report_job_result(log, result)
        # A submission that was just packaged should be selectable without
        # leaving and re-entering the screen. This block spent a while on
        # RunScreen by mistake, where the condition could never be true and
        # the method did not exist -- so packaging appeared to succeed
        # (exit 0, folder written) while the picker still read "Nothing
        # packaged yet".
        if ok and "community-package" in argv:
            self._refresh_pickers()


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
    """Browses reports `cbench` already wrote -- never generates or edits
    one.

    Leads with a TABLE of the reports that exist (model, suite, kind,
    date), because that is how someone looks for one: "the channel report
    for qwen3 from Tuesday", not "results/s2_channel/trial_summary_qwen3-
    0.6b.md". The directory tree is still there underneath for anything
    the table doesn't classify, but nobody should have to navigate it to
    find a report this screen could have listed."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    # Report filenames this framework writes, most specific first:
    #   trial_summary_<tag>.md        -- N trials pooled (the citable one)
    #   <suite>_report_<tag>.md       -- one run of one suite
    #   scorecards/<tag>.md           -- the cross-suite grade
    # The model tag is whatever remains once the prefix is removed, which
    # is why these are matched rather than split on "_": a tag itself
    # contains underscores often enough that splitting mangles it.
    REPORT_PREFIXES = (
        ("trial_summary_", "trial summary"),
        ("containment_report_", "S1 single run"),
        ("channel_report_", "S2 single run"),
        ("persistence_report_", "S3 single run"),
    )
    SUITE_LABEL = {
        "s1_containment": "S1 containment",
        "s2_channel": "S2 channel",
        "s3_persistence": "S3 persistence",
        "scorecards": "scorecard",
        "tui-logs": "run log",
    }

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        root = _results_root()
        with Vertical(id="reports-form"):
            yield Static(
                f"Reports in [bold]{root}[/bold] -- trial summaries and scorecards. "
                f"Pick a row to read it. Newest first. Raw CSVs and run logs are in the "
                f"tree below.",
                id="reports-caption",
            )
            with Horizontal(id="reports-body"):
                with Vertical(id="reports-left"):
                    yield DataTable(id="reports-table")
                    if root.exists():
                        yield DirectoryTree(str(root), id="reports-tree")
                    else:
                        yield Static(
                            f"No results directory at {root}.\n\n"
                            f"If you have run suites before, you are almost certainly in a "
                            f"different working directory than when you ran them -- results/ "
                            f"is resolved relative to where you launch cbench. cd to your "
                            f"project directory and reopen, or set "
                            f"$OPENLLM_CBENCH_RESULTS_DIR.",
                            id="reports-empty",
                        )
                with VerticalScroll(id="reports-viewer-container"):
                    yield TextArea("", id="reports-viewer", read_only=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#reports-table", DataTable)
        table.add_columns("Model", "Suite", "Kind", "Date", "File")
        table.cursor_type = "row"
        self._paths = {}
        self._load_reports()

    @staticmethod
    def _strip_timestamp(tag):
        """Drops a trailing _YYYYmmdd_HHMMSS. A single-run report carries
        one; the run time belongs in the Date column, not glued to the
        model name where it stops the column being scannable."""
        return re.sub(r"_\d{8}_\d{6}$", "", tag)

    def _classify(self, path: Path, root: Path):
        rel = path.relative_to(root)
        suite = self.SUITE_LABEL.get(rel.parts[0], rel.parts[0]) if len(rel.parts) > 1 else "-"
        stem = path.stem

        for prefix, kind in self.REPORT_PREFIXES:
            if stem.startswith(prefix):
                return self._strip_timestamp(stem[len(prefix):]), suite, kind
        if rel.parts[0] == "scorecards":
            return stem, suite, "scorecard"
        return stem, suite, "report"

    @work(exclusive=True)
    async def _load_reports(self) -> None:
        import asyncio
        import datetime

        from textual.css.query import NoMatches

        root = _results_root()
        if not root.exists():
            return

        def scan():
            # REPORTS, not every artefact. A results tree holds hundreds of
            # per-trial CSVs and run logs; listing them all buried the six
            # documents someone actually opens (261 rows on this machine,
            # of which 6 were readable reports). Raw CSVs and logs are
            # still reachable through the tree below -- they are inputs
            # and evidence, not things you browse.
            rows = []
            for pth in root.rglob("*.md"):
                if pth.is_file():
                    rows.append((pth, pth.stat().st_mtime))
            return sorted(rows, key=lambda r: r[1], reverse=True)

        found = await asyncio.to_thread(scan)
        try:
            table = self.query_one("#reports-table", DataTable)
        except NoMatches:
            return

        for pth, mtime in found:
            model, suite, kind = self._classify(pth, root)
            when = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            key = table.add_row(model, suite, kind, when, pth.name)
            self._paths[key] = pth

        try:
            self.query_one("#reports-caption", Static).update(
                f"{len(found)} report(s) in [bold]{root}[/bold] -- trial summaries and "
                f"scorecards. Pick a row to read it. Newest first. Raw CSVs and run logs "
                f"are in the tree below."
            )
        except NoMatches:
            pass

    def _show(self, path: Path) -> None:
        viewer = self.query_one("#reports-viewer", TextArea)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > 200_000:
                text = text[:200_000] + "\n... [truncated for display]"
            viewer.text = text
        except Exception as e:
            viewer.text = f"Could not read {path}: {e}"

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        path = self._paths.get(event.row_key)
        if path is not None:
            self._show(path)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        path = self._paths.get(event.row_key)
        if path is not None:
            self._show(path)

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self._show(Path(event.path))


class CBenchTUI(App):
    TITLE = "openllm-cbench"
    BINDINGS = [("q", "quit", "Quit")]

    CSS = """
    InvariantBar { background: $warning-darken-2; color: $text; padding: 0 1; }
    #dashboard-buttons, #dashboard-buttons-2 { height: auto; }
    #dashboard-buttons Button, #dashboard-buttons-2 Button { margin: 0 1 1 0; }
    #doctor-caption { color: $text-muted; padding: 0 0 1 0; }
    #run-form, #gate-form, #models-body, #pull-form, #score-form, #community-form { padding: 1; }
    RichLog { height: 1fr; border: solid $accent; }
    #reports-body { height: 1fr; }
    #reports-tree { width: 40%; }
    #reports-viewer-container { width: 60%; }
    #models-buttons { height: auto; }
    #models-buttons Button { margin: 0 1 0 0; }
    #models-table { height: 12; border: solid $accent; }
    /* Both the tree and its empty-state placeholder need the same width.
       Without this the placeholder (shown whenever community-results/
       doesn't exist yet -- i.e. on a fresh install, the common case)
       expands to fill the row and pushes the action buttons off the right
       edge of the terminal, where they cannot be clicked at all. */
    #community-tree, #community-tree-empty { width: 40%; }
    #community-form-inner { width: 60%; }
    #score-suite-checks { height: auto; }
    #score-suite-checks Checkbox { margin: 0 2 0 0; }
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
