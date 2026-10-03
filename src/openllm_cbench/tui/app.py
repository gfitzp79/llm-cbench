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

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
# Module level: ten teardown guards below catch it, and were written
# against a name that was only ever imported inside two functions -- so
# each guard raised NameError instead of returning quietly.
from textual.css.query import NoMatches
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button, Checkbox, DataTable, DirectoryTree, Footer, Header, Input, ProgressBar, RichLog,
    Select, Static, TextArea,
)

from openllm_cbench.core import exitcodes
from openllm_cbench.core.invariant import SAFETY_INVARIANT
from openllm_cbench.core.plan import is_row_line, parse_total
from openllm_cbench.core.runclock import time_from_filename
from openllm_cbench.tui.jobs import (
    RUNNABLE_SUITES, build_args, cbench_command, condensed_line_filter, parse_trial_header,
    run_job, save_job_log,
)


def _results_root() -> Path:
    # THE resolver, not a copy of it. A copy here once skipped the config
    # file, so after pinning a location under Settings the dashboard showed
    # the pinned folder while Browse reports listed ./results and found
    # nothing.
    from openllm_cbench.core.paths import results_root
    return results_root()


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
        _fill([], "Could not list local models: type the tag below")
        return {}
    options = [(m["name"], m["name"]) for m in sorted(local, key=lambda x: x["name"])]
    _fill(options, "Choose a local model, or type a tag below" if options else
          "No local models found: type the tag below")
    return {m["name"]: m for m in local}


def _saved_log_line(result) -> str:
    """Where the full log went, or that it could not be saved -- never
    "saved to None". save_job_log() returns None rather than raising when
    the results folder cannot be written."""
    path = save_job_log(result)
    if path is None:
        return ("[yellow]Could not save the full log: the results folder could not be "
                "written. The output above is all there is.[/yellow]")
    return f"[dim]Full log saved to {path}[/dim]"


def _refusal_reason(lines) -> str:
    """The reason a refused command gave, from its own "[!] NOT STARTING:"
    line, or "" when it printed none."""
    for line in reversed(lines or []):
        text = line.strip()
        if text.startswith("[!] NOT STARTING"):
            return text[len("[!] NOT STARTING"):].lstrip(": ").strip()
    return ""


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
    log.write(_saved_log_line(result))
    if result.error:
        log.write(f"[bold red]{result.error}[/bold red]")
        return False
    # The last line on screen is the most prominent one, so it says what
    # happened rather than what number the process returned. See
    # core/exitcodes.py -- "exit code: 2" was reported as meaning nothing,
    # and it did.
    ok = exitcodes.is_success(result.returncode)
    style = "bold green" if ok else "bold red"
    # A refusal's reason is printed near the top of the output, and the log
    # box shows the last few lines: "the reason is printed above" pointed at
    # text that had scrolled out of view. Repeat it here, next to the verdict.
    reason = _refusal_reason(result.lines) if result.returncode == exitcodes.REFUSED else ""
    if reason:
        log.write(f"[bold red]Why: {escape(reason)}[/bold red]")
    meaning = exitcodes.describe(result.returncode,
                                 exitcodes.subcommand_of(result.argv))
    log.write(f"[{style}]{meaning}[/{style}]")
    if ok and success_note:
        log.write(f"[dim]{success_note}[/dim]")
    return ok


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
                yield Button("Check environment", id="run-doctor")
                yield Button("Settings", id="goto-settings")
                yield Button("About / extend this", id="goto-about")
            yield Static("", id="results-location")
            yield Static("", id="progress-counts")
            yield Static("", id="progress-tier")
            yield Static("", id="progress-next")
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
        self._show_results_location()
        self._load_progress()
        # Deliberately does NOT run `cbench doctor` automatically -- every
        # cbench action this TUI takes happens because of a click, none on
        # app startup, so what ran and why is never ambiguous.
        self.query_one("#doctor-log", RichLog).write(
            "[dim]Press \"Check environment\" above to run it.[/dim]"
        )

    def _show_results_location(self) -> None:
        """Where results are going, on screen before anything runs.

        The TUI gets launched from whatever shell is open, so the
        directory that used to decide this was usually one nobody chose.
        A run that lands in a second tree is not just hard to find --
        `cbench score` globs per directory, so it is silently missing from
        the aggregate."""
        from openllm_cbench.core.config import resolution

        try:
            widget = self.query_one("#results-location", Static)
        except NoMatches:
            return
        path, source, pinned = resolution()
        if pinned:
            widget.update(f"[dim]Results:[/dim] [bold]{path}[/bold] [dim]({source})[/dim]")
        else:
            widget.update(
                f"[bold yellow]Results:[/bold yellow] [bold]{path}[/bold] "
                f"[yellow](not pinned, so this follows whichever directory the TUI "
                f"was launched from). Settings fixes it.[/yellow]")

    @work(exclusive=True, group="progress")
    async def _load_progress(self) -> None:
        """Reads local files plus one endpoint call for the model list.

        Everything here is read-only: no model is called, nothing is
        written. The endpoint list is the only network touch, and failing
        it reports "unreachable" rather than zero -- "we could not ask"
        and "you have none" are different facts."""
        # Local import, matching this module's own convention -- asyncio is
        # imported per-function here, not at module level.
        import asyncio

        from openllm_cbench.core.progress import compute_progress, render_line

        def gather():
            try:
                from openllm_cbench.core.discover import list_local_models
                local = list_local_models()
            except Exception:
                local = None
            return compute_progress(local_models=local)

        try:
            progress = await asyncio.to_thread(gather)
        except Exception as exc:
            # Say so rather than leaving the panel blank. A blank panel is
            # indistinguishable from "you have done nothing", which is a
            # different and discouraging claim.
            try:
                self.query_one("#progress-counts", Static).update(
                    f"[dim]Could not read local progress: {exc}[/dim]")
            except NoMatches:
                pass
            return

        # Same teardown discipline as every other worker here: the screen
        # can be gone by the time this resolves.
        try:
            counts = self.query_one("#progress-counts", Static)
            tier = self.query_one("#progress-tier", Static)
            nxt = self.query_one("#progress-next", Static)
        except NoMatches:
            return

        counts.update(f"[bold]{render_line(progress)}[/bold]")
        tier.update(
            f"Level: [bold]{progress['tier']}[/bold]  [dim](earned on rigour, not volume: "
            f"a result counts as [b]citable[/b] when the model is gate-checked, no suite was "
            f"refused by a validity guard, and every suite ran at least "
            f"{progress['min_citable_trials']} trials)[/dim]"
        )
        action = progress.get("next_action")
        if action:
            nxt.update(f"[bold]Next:[/bold] {action[0]}. [dim]{action[1]}[/dim]")
        else:
            nxt.update("")


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
        elif event.button.id == "run-doctor":
            self.run_doctor()
        elif event.button.id == "goto-settings":
            self.app.push_screen(SettingsScreen())
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
        result = await run_job(argv, on_line=lambda line: log.write(line), live_log=True)
        log.write(_saved_log_line(result))


class RunScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        with Vertical(id="run-form"):
            yield Static("Run a suite: equivalent to running `cbench <suite>` yourself. "
                         "One run of one suite: writes a CSV and a report, no grade. For a "
                         "grade, use Score a model.")
            yield Select(
                [(label, subcmd) for label, subcmd in RUNNABLE_SUITES],
                id="suite-select", value=RUNNABLE_SUITES[0][1], allow_blank=False,
            )
            yield Select([], id="model-select", allow_blank=True,
                         prompt="Choose a local model. Loading...")
            yield Input(placeholder="or type a model tag, e.g. gemma3:12b", id="model-input")
            # OFF by default, matching the CLI's own store_true default.
            yield Checkbox(
                "Preview only: show the request without calling the model",
                id="dry-run-checkbox", value=False,
            )
            yield Input(
                # The old example (--boundary/--sandbox) exists only on
                # containment, so it errored on the other two suites.
                placeholder="extra flags for this suite (optional), e.g. --think false "
                            "(see `cbench <suite> --help`)",
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
        result = await run_job(argv, on_line=lambda line: log.write(line), live_log=True)
        _report_job_result(log, result)


class GateScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        with Vertical(id="gate-form"):
            yield Static("Gate-check a model: equivalent to running `cbench gate` yourself.")
            yield Select([], id="gate-model-select", allow_blank=True,
                         prompt="Choose a local model. Loading...")
            yield Input(placeholder="or type a model tag, e.g. gemma3:12b", id="gate-model-input")
            yield Checkbox(
                "Save the result to my catalogue (unticked: show it only, write nothing)",
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
        result = await run_job(argv, on_line=lambda line: log.write(line), live_log=True)
        _report_job_result(log, result)


def _short_date(iso):
    """`2026-09-18T10:07:54.359+01:00` -> `2026-09-18`. Returns "" for a
    value that isn't a timestamp rather than showing a mangled one."""
    if not isinstance(iso, str) or len(iso) < 10:
        return ""
    head = iso[:10]
    return head if head[4] == "-" and head[7] == "-" else ""


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return -1.0


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
                "model catalogue already has config guidance for this exact tag; "
                "an uncatalogued model still runs fine, just \"ungated\".\n"
                "Speed column: a bold tok/s figure is MEASURED on this machine during that "
                "model's gate check. A word (fast/good/moderate/slow) is an estimate from "
                "the parameters actually read per token: for a mixture-of-experts model "
                "that's the active experts, not the full weight count, which is why a "
                "30B-A3B MoE outruns a dense 30B of the same size on disk.\n"
                "Score column: an A-F grade (0-100), the worst of the three suites "
                "run, not an average (see \"Scoring a model\" in docs/USER_GUIDE.md). "
                "\"not scored\" = never run through `cbench score`. \"needs re-score\" = "
                "scored under older scoring rules, so the grade is hidden: re-score it on "
                "the Score screen with \"Re-score saved results only\" ticked, which reads "
                "the saved CSVs and calls no model. \"N/A\" = nothing "
                "gradable yet. \"INVALID\" = a validity guard fired (e.g. saved runs that "
                "asked different tasks or probes, disagree on sampling or generation "
                "budgets, or mix runs that recorded those settings with older ones that "
                "did not; move the older runs out of the results folder). "
                "That suite is excluded from the grade; do not trust it yet. Trailing \"*\" = an "
                "otherwise-ok suite still has an unresolved caveat: read the full "
                f"scorecard ({escape(str(_results_root() / 'scorecards'))}/<tag>.md) before "
                "citing the grade alone.",
                id="models-score-legend",
            )
            with Horizontal(id="models-limit-row"):
                # `cbench discover --gate-all` has had a --limit since it
                # was written, and no way to reach it from here. Gating
                # every uncatalogued model is a real model call per model
                # and can run for an hour on a full Ollama library; the
                # bound belongs in front of the button that starts it,
                # not in --help.
                yield Static("Gate at most", id="models-limit-label")
                yield Input(placeholder="all", id="models-limit-input",
                            value="", type="integer")
                yield Static("model(s) per batch", id="models-limit-suffix")
            with Horizontal(id="models-delete-row"):
                # The one destructive action in this app. The checkbox is
                # deliberately not remembered between uses: an armed
                # delete sitting around from five minutes ago is how the
                # wrong model gets removed.
                yield Checkbox(
                    "Confirm delete: permanently removes the selected model's "
                    "weights from your endpoint (results are kept)",
                    id="models-confirm-delete", value=False,
                )
                yield Button("Delete selected", id="models-delete", variant="error")
            with Horizontal(id="models-buttons"):
                yield Button("Refresh", id="models-refresh", variant="primary")
                yield Button("Gate + save selected", id="models-gate-selected")
                yield Button("Gate + save all uncatalogued", id="models-gate-all")
                yield Button("Search / pull a new model", id="goto-pull")
                yield Button("Back", id="models-back")
            yield DataTable(id="models-table")
            yield RichLog(id="models-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    _rows: list = []
    _sort_by: str = None
    _sort_desc: bool = False

    def on_mount(self) -> None:
        self._rows = []
        table = self.query_one("#models-table", DataTable)
        table.add_columns("Tag", "Params (B)", "Quant", "Size", "Added", "Fit",
                          "Speed", "Catalogued", "Score")
        # Clicking a header sorts by it. The rows carry parallel sort keys
        # (see _rows) because sorting the RENDERED cell would sort
        # "9.0 GB" after "10.5 GB" as strings, and markup like
        # "[green]fits[/green]" by its escape code.
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
        elif event.button.id == "models-delete":
            self._delete_selected()

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

    def on_data_table_header_selected(self, event) -> None:
        """Click a header to sort by it; click the same one again to
        reverse. Sorts the parallel key, never the rendered cell -- the
        cell is formatted for a human and sorting it puts "9.0 GB" after
        "10.5 GB" and orders Fit by the colour of its markup."""
        if event.data_table.id != "models-table":
            return
        label = str(event.column_key.value or event.label)
        self._sort_desc = (label == self._sort_by) and not self._sort_desc
        self._sort_by = label
        self._apply_sort()

    def _apply_sort(self, quiet: bool = False) -> None:
        """Re-renders the rows in sorted order.

        Rebuilds rather than calling DataTable.sort(), because that hands
        its key function the ROW VALUES -- the formatted cells. Sorting
        those puts "9.0 GB" after "10.5 GB" and orders Fit by the colour
        of its markup. The raw values were kept beside each row precisely
        so this never has to read one back out."""
        if not self._sort_by or not self._rows:
            return
        try:
            table = self.query_one("#models-table", DataTable)
        except NoMatches:
            return
        col = self._sort_by

        def key_for(item):
            v = (item[1] or {}).get(col, "")
            if v is None:
                return (1, "")
            # One comparable type per column: a str never meets a float.
            return (0, v.lower()) if isinstance(v, str) else (0, v)

        try:
            ordered = sorted(self._rows, key=key_for, reverse=self._sort_desc)
        except TypeError:
            return
        self._rows = ordered
        table.clear()
        for cells, _ in ordered:
            table.add_row(*cells)
        if quiet:
            return
        arrow = "descending" if self._sort_desc else "ascending"
        try:
            self.query_one("#models-log", RichLog).write(
                f"[dim]Sorted by {col}, {arrow}. Click the same header again to "
                f"reverse it.[/dim]")
        except NoMatches:
            pass

    def _selected_tag(self):
        """The tag under the cursor, or None. Reads column 0 of the row
        rather than an index into anything else, so adding a column
        cannot silently change which value this returns."""
        try:
            table = self.query_one("#models-table", DataTable)
        except NoMatches:
            return None
        if table.row_count == 0 or table.cursor_row is None:
            return None
        try:
            return str(table.get_row_at(table.cursor_row)[0])
        except Exception:
            return None

    def _delete_selected(self) -> None:
        """`cbench remove --model <tag> --yes`, the one destructive action
        this app can take.

        Three guards, all of them cheap and all of them earned: a row must
        actually be selected, the confirmation must be ticked in this same
        interaction, and the exact tag is echoed before the subprocess
        runs so what is about to happen is on screen in words."""
        log = self.query_one("#models-log", RichLog)
        tag = self._selected_tag()
        if not tag:
            log.write("[bold red]Select a model row first: nothing is selected, so "
                      "there is nothing to delete.[/bold red]")
            return
        if not self.query_one("#models-confirm-delete", Checkbox).value:
            log.write(
                f"[bold yellow]NOT deleting '{tag}'.[/bold yellow] Deleting removes the "
                f"model's weights from your endpoint and cannot be undone from here: "
                f"getting it back means pulling it again. Tick the confirm box if that "
                f"is what you want.")
            return

        argv = cbench_command("remove", ["--model", tag, "--yes"])
        log.write(f"[dim]$ {' '.join(argv)}[/dim]")
        log.write(f"[bold red]Deleting '{tag}' from the endpoint.[/bold red] "
                  f"[dim]Its CSVs, reports and scorecard stay in results/. The "
                  f"measurement outlives the weights.[/dim]")
        # Untick immediately, so the next delete has to be confirmed on its
        # own terms rather than inheriting this one.
        self.query_one("#models-confirm-delete", Checkbox).value = False
        self._delete_worker(argv, log)

    @work(exclusive=True, group="models-delete")
    async def _delete_worker(self, argv, log) -> None:
        result = await run_job(argv, lambda line: log.write(line))
        if result.returncode == 0:
            log.write("[green]Deleted. Refreshing the list...[/green]")
            self._refresh()
        else:
            log.write(f"[bold red]Delete failed, nothing was removed: "
                      f"{exitcodes.describe(result.returncode, 'remove')}[/bold red]")

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
        self._rows = []
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

            added_raw = m.get("modified_at") or ""
            cells = (
                m["name"], str(m["params_b"]), m["quant"], format_size(m["size"]),
                _short_date(added_raw), fit_cell,
                speed_cell,
                "yes" if is_cat else "[bold yellow]no[/bold yellow]",
                score if score != "not scored" else "[dim]not scored[/dim]",
            )
            table.add_row(*cells)
            # Sort keys kept beside the row rather than parsed back out of
            # the rendered cells. A displayed value is for a human; sorting
            # it is how "9.0 GB" ends up after "10.5 GB".
            self._rows.append((cells, {
                "Tag": m["name"].lower(),
                "Params (B)": _as_float(m.get("params_b")),
                "Quant": (m.get("quant") or "").lower(),
                "Size": m.get("size") or 0,
                "Added": added_raw,
                "Fit": {"fits": 0, "tight": 1, "spills": 2, "unknown": 3}[assessment["tier"]],
                "Speed": -(perf.get("tok_s") or 0) if perf["source"] == "measured" else
                         {"fast": 1, "good": 2, "moderate": 3, "slow": 4, "unknown": 5}[perf["label"]] + 100,
                "Catalogued": 0 if is_cat else 1,
                "Score": score or "",
            }))
        if self._sort_by:
            self._apply_sort(quiet=True)
        if spills_n:
            log.write(f"[bold yellow]{spills_n} model(s) won't fit in this GPU's "
                       f"{vram_mb:,} MB of VRAM and will run partly on CPU: much slower, but "
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
            log.write("[bold red]No row selected: click a model first.[/bold red]")
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
        flags = ["--gate-all"]
        raw = self.query_one("#models-limit-input", Input).value.strip()
        if raw:
            try:
                limit = int(raw)
            except ValueError:
                limit = 0
            if limit > 0:
                flags += ["--limit", str(limit)]
            else:
                # Refuse rather than widen. A typo in a field meant to LIMIT
                # model calls must not turn into calling every model.
                log.write("[bold red]Limit must be a positive whole number, or blank for "
                          "all; nothing started.[/bold red]")
                return
        argv = cbench_command("discover", flags)
        log.write(f"[dim]$ {' '.join(argv)}[/dim]")
        log.write("[dim]Gate-checking every uncatalogued model, one at a time: real model "
                  "calls, so this can take a while for a long list.[/dim]")
        self._gate_worker(argv, log)

    @work(exclusive=True)
    async def _gate_worker(self, argv, log: RichLog) -> None:
        result = await run_job(argv, on_line=lambda line: log.write(line), live_log=True)
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
                "pull it if you want it: equivalent to running `cbench search` / "
                "`cbench pull` (or `ollama pull`) yourself. [bold]Pull downloads real "
                "data from Ollama's registry[/bold], possibly several GB, and can take "
                "a while; Search never does."
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
        result = await run_job(argv, on_line=lambda line: log.write(line), live_log=True)
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
        result = await run_job(argv, on_line=lambda line: log.write(line), live_log=True)
        _report_job_result(log, result,
                            "Done. Go to \"Local models\" and gate + save it to add it "
                            "to your catalogue.")


class ScoreScreen(Screen):
    """`cbench score` -- a real subprocess, same as every other
    action-taking screen. All the actual depth->trials mapping, the
    scorecard computation, and the run-lock live in cli.py's _cmd_score()
    and scoring/scorecard.py, not here; this screen is a form and a log.
    The resulting scorecard is what the Local models screen's Score column
    (ModelsScreen) reads afterward -- run this first, then check there or
    in "Browse reports"."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        # Every label says what ticking it does, in the user's words, and
        # names no flag: "the language doesn't read clean, not clear on what
        # the user has to do" (owner, 2026-09-23). The preview line under
        # the buttons still shows the exact command.
        with VerticalScroll(id="score-form"):
            yield Static(
                "Runs the suites against one model and saves an A-F grade. The worst "
                "suite sets the grade. Grades appear on the Local models screen and in "
                "`cbench catalogue`.",
                id="score-intro",
            )
            yield Static("[b]Model[/b]", classes="score-heading")
            yield Select([], id="score-model-select", allow_blank=True,
                         prompt="Choose a local model. Loading...")
            yield Input(placeholder="or type a model tag, e.g. gemma3:12b",
                        id="score-model-input")
            yield Static("", id="score-catalogue-status")
            yield Static("", id="score-hardware-status")
            yield Static("[b]Suites[/b]", classes="score-heading")
            with Horizontal(id="score-suite-checks"):
                yield Checkbox("S1 containment", id="score-s1", value=True)
                yield Checkbox("S2 channel", id="score-s2", value=True)
                yield Checkbox("S3 persistence", id="score-s3", value=True)
            yield Static("", id="score-suite-note")
            yield Static("[b]Depth[/b]", classes="score-heading")
            yield Select(
                [("Quick: 1 trial per suite (2 for S3). A first look, not a grade to cite",
                  "quick"),
                 ("Standard: 3 trials per suite. The minimum for a grade worth citing",
                  "standard"),
                 ("Thorough: 5 trials per suite", "thorough")],
                id="score-depth-select", value="standard", allow_blank=False,
            )
            yield Static("[b]Generation budget[/b] (optional)", classes="score-heading")
            yield Static(
                "[dim]Leave both blank: cbench sizes them to the model, with a larger "
                "budget for a model that reasons. Enter a number only to override that.[/dim]",
                id="score-budget-help",
            )
            with Horizontal(id="score-budget-row"):
                # Blank means automatic (core/budget.py). An explicit value
                # beats the catalogue's config_overrides, which beat the
                # automatic value: the same precedence the CLI flags have.
                yield Input(placeholder="Context window in tokens (blank: automatic)",
                            id="score-num-ctx")
                yield Input(placeholder="Reply limit in tokens (blank: automatic)",
                            id="score-num-predict")
            yield Static("[b]Options[/b]", classes="score-heading")
            yield Checkbox(
                "Gate-check the model first if it is not in your catalogue (recommended)",
                id="score-gate-first", value=True,
            )
            yield Checkbox(
                "Re-score saved results only (runs nothing and calls no model)",
                id="score-from-existing", value=False,
            )
            # OFF by default, matching `cbench score`, whose --dry-run is
            # store_true. It defaulted ON here once, so the button labelled
            # "Score" did not score unless you noticed and unticked it.
            yield Checkbox(
                "Preview only: show the requests without calling the model",
                id="score-dry-run", value=False,
            )
            yield Checkbox(
                "Also run suites that cannot measure this model (they grade INVALID; "
                "only useful for their transcripts)",
                id="score-force-uncheckable", value=False,
            )
        # Outside the scrolling form, so the button, the sentence saying what
        # it will do, and the run's output stay on screen however far the
        # form is scrolled. The log had whatever height the form left over,
        # a few lines, and a refusal's reason scrolled out of it.
        with Vertical(id="score-run"):
            with Horizontal(id="score-buttons"):
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
        self._refresh_preview()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "score-model-input":
            self._refresh_catalogue_status(event.value.strip())
        self._refresh_preview()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        self._refresh_preview()

    def _score_flags(self):
        """What this form currently describes, as (argv-tail, trials, problems).

        Pure: reads the widgets, writes nothing, touches no disk. That is
        what lets the live preview and the actual launch share it -- a
        preview built from a second copy of this logic is a preview that
        can lie about what the button will do, which is the failure being
        fixed here, not a new one to introduce."""
        model = self.query_one("#score-model-input", Input).value.strip()
        suites = [s for s in ("s1", "s2", "s3")
                  if self.query_one(f"#score-{s}", Checkbox).value]
        problems = []
        if not model:
            problems.append("A model tag is required.")
        if not suites:
            problems.append("Select at least one suite.")

        args = ["--model", model, "--suites", ",".join(suites)]
        total_trials = None
        if self.query_one("#score-from-existing", Checkbox).value:
            args.append("--from-existing")
        else:
            depth = self.query_one("#score-depth-select", Select).value
            args += ["--depth", str(depth)]
            # Budgets are RECORDED in every row, so a run at a non-default
            # budget stays comparable-or-refused rather than silently
            # pooling with one at a different budget.
            for widget_id, flag in (("#score-num-ctx", "--num-ctx"),
                                    ("#score-num-predict", "--num-predict")):
                raw = self.query_one(widget_id, Input).value.strip()
                if raw:
                    try:
                        args += [flag, str(int(raw))]
                    except ValueError:
                        problems.append(f"{flag} must be a whole number, got {raw!r}; "
                                        f"ignoring it.")
            if self.query_one("#score-dry-run", Checkbox).value:
                args.append("--dry-run")
            # Without this the TUI has no way past the pre-flight refusal
            # at all, which for a model where every suite is dead means
            # being stopped with the fix named only as a CLI flag the
            # screen cannot send.
            if self.query_one("#score-force-uncheckable", Checkbox).value:
                args.append("--force-uncheckable")
            from openllm_cbench.scoring.scorecard import depth_trials
            if suites:
                total_trials = sum(depth_trials(depth, s) for s in suites)
        return args, total_trials, problems

    def _gate_first_argv(self, args):
        """The `cbench gate` that will run before `score`, or None.

        Shared by the preview and the launch, like _score_flags. It lived
        only in the launch once, so a dry run -- previewed as "calls no
        model" -- gate-checked an uncatalogued model and wrote the result
        into the catalogue anyway. A dry run and --from-existing both
        promise no model call, and a gate check is one, so both skip it."""
        if "--dry-run" in args or "--from-existing" in args:
            return None
        if not self.query_one("#score-gate-first", Checkbox).value:
            return None
        model = args[args.index("--model") + 1]
        if not model:
            return None
        from openllm_cbench.core.registry import load_registry, lookup
        if lookup(model, load_registry()) is not None:
            return None
        return cbench_command("gate", ["--model", model, "--save"])

    def _refresh_preview(self) -> None:
        """Say in words what pressing Score will do, before it is pressed.

        Reported live: "UI is unintuitive, I'm not sure if I need to check
        or uncheck the options". Checkbox glyphs render near-identically in
        some terminals, so the state of a tick is not readable from the
        tick. A sentence naming the consequence is, and it does not depend
        on a font."""
        try:
            preview = self.query_one("#score-preview", Static)
        except Exception:
            return
        try:
            args, total_trials, problems = self._score_flags()
        except Exception:
            # The form is mid-construction (on_mount fires handlers before
            # every widget exists). A preview is not worth an exception.
            return

        # A blocking problem means there is no command to describe. A
        # non-blocking one (a typo'd budget, which is dropped) still has a
        # run behind it, so the warning goes ABOVE the command rather than
        # replacing it -- hiding what will run is the thing being fixed.
        blocking = [p for p in problems if "required" in p or "at least one" in p]
        if blocking:
            preview.update("[yellow]" + "  ".join(blocking) + "[/yellow]")
            return
        warning = ("[yellow]" + "  ".join(problems) + "[/yellow]\n") if problems else ""

        if "--from-existing" in args:
            what = ("Will re-score the results already saved for this model. Runs no "
                    "trials and calls no model.")
        elif "--dry-run" in args:
            what = ("PREVIEW ONLY: shows the requests and calls no model. Nothing is "
                    "saved. Untick \"Preview only\" to score.")
        else:
            what = f"Will run {total_trials} trial(s) against the model and save a grade."
            if "--force-uncheckable" in args:
                what += (" Suites that cannot measure this model will run anyway, take "
                         "their full time, and grade INVALID.")
        argv = cbench_command("score", args)
        gate_argv = self._gate_first_argv(args)
        first = ""
        if gate_argv:
            first = ("[b]First: gate-check this model, which is not in your catalogue "
                     "(calls the model and saves the result to your catalogue).[/b]\n"
                     f"[dim]$ {' '.join(gate_argv)}[/dim]\n")
        preview.update(f"{warning}{first}[b]{what}[/b]\n[dim]$ {' '.join(argv)}[/dim]")

    def _refresh_catalogue_status(self, model: str) -> None:
        """Local-only registry read (same function `cbench discover`/
        `cbench catalogue` already call), synchronous -- reading two small
        JSON files off disk doesn't need a worker the way a live endpoint
        call does. Shows catalogue status BEFORE a run starts, not just in
        the log after one is already underway -- a model this catalogue
        has never seen is the common case for anyone testing a new model,
        not an edge case worth discovering only mid-run.

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
            status.update("[green]✓ In your catalogue[/green]: gated before, so the "
                          "settings it needs are known.")
        else:
            status.update(
                "[bold yellow]⚠ Not in your catalogue[/bold yellow]: it will be gate-checked "
                "first, unless you untick that option below."
            )
        hardware.update(self._hardware_fit_line(model, entry))
        self._apply_suite_readiness(model, entry)

    # Why a suite is unticked, in a sentence rather than the gate's full
    # diagnosis (that is in the gate report and the catalogue entry).
    _SUITE_OFF_REASON = {
        "s1": "S1 needs working tool calls, and this model's did not work when it was gated.",
        "s3": "S3 needs working tool calls, and this model's did not work when it was gated.",
    }

    def _apply_suite_readiness(self, model, entry) -> None:
        """Ticks the suites that can measure this model and unticks the rest,
        from what `cbench gate` recorded in the catalogue, and says why.

        All three were ticked for every model, so a model the gate had
        already found unable to run a suite went to the pre-flight with it
        ticked and was refused there, its reason scrolled out of view.
        Applied only when the model changes, so a box the user ticks back
        stays ticked; the pre-flight still skips a suite that cannot run."""
        if model == getattr(self, "_suites_set_for", None):
            return
        self._suites_set_for = model
        from openllm_cbench.core.preflight import TOOL_SUITES
        readiness = (entry or {}).get("suite_readiness") or {}
        off = []
        try:
            for suite in ("s1", "s2", "s3"):
                # Only a tool suite can be unable to run. An S2 verdict of
                # INVALID in the catalogue predates S2 grading probe failure
                # on a model without a reasoning channel, so it is ignored.
                verdict = (readiness.get(suite) or [""])[0] if suite in TOOL_SUITES else ""
                can_run = verdict != "invalid"
                self.query_one(f"#score-{suite}", Checkbox).value = can_run
                if not can_run:
                    off.append(suite)
            note = self.query_one("#score-suite-note", Static)
        except NoMatches:
            return
        if off:
            note.update("[yellow]" + " ".join(self._SUITE_OFF_REASON[s] for s in off)
                        + f" {'It is' if len(off) == 1 else 'They are'} unticked; the grade "
                        f"covers the rest.[/yellow]")
        else:
            note.update("")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "score-back":
            self.app.pop_screen()
        elif event.button.id == "score-start":
            self._start_score()

    def _start_score(self) -> None:
        model = self.query_one("#score-model-input", Input).value.strip()
        log = self.query_one("#score-log", RichLog)
        log.clear()

        # One source of truth for what this form means, shared with the
        # live preview -- see _score_flags().
        args, total_trials, problems = self._score_flags()
        blocking = [p for p in problems if "required" in p or "at least one" in p]
        for problem in problems:
            log.write(f"[bold red]{problem}[/bold red]")
        if blocking:
            return

        suites = args[args.index("--suites") + 1].split(",")
        from_existing = "--from-existing" in args

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
                    f"[bold red]\"Re-score saved results only\" is ticked, but no "
                    f"{'/'.join(s.upper() for s in suites)} results are saved yet for '{model}': "
                    f"there is nothing to re-score, and the grade would be N/A. Untick it "
                    f"(and pick a depth) to run the suites.[/bold red]"
                )
                return
            elif missing:
                log.write(
                    f"[bold yellow]Re-score saved results only: no {'/'.join(s.upper() for s in missing)} CSVs "
                    f"exist yet for '{model}', so {'those suites' if len(missing) > 1 else 'that suite'} will "
                    f"show 'not run' below; the grade will only reflect whichever suite(s) do have "
                    f"data.[/bold yellow]"
                )

        argv = cbench_command("score", args)
        self._refresh_preview()
        log.write(f"[dim]$ {' '.join(argv)}[/dim]")

        progress = self.query_one("#score-progress", ProgressBar)
        if total_trials:
            # Starts in trials (suites x trials-for-depth), and switches to
            # rows when the run prints its `Plan:` line (see _run_worker).
            # Trials alone were measured to be the wrong unit: one quick
            # score's trials took 2:39, 11:54, 0:33 and 0:29, so the ETA
            # was blank for the whole first trial, then a third of the
            # real time, then frozen, because Textual's ProgressBar
            # projects only 30 seconds past its last update. Rows finish
            # every few seconds. The ETA itself is still Textual's, from
            # the observed rate of .advance() calls.
            progress.update(total=total_trials, progress=0)
            progress.display = True
        else:
            progress.display = False

        # Gate first if this tag has never been seen and the checkbox says
        # to -- a convenience, not a requirement: this framework never
        # refuses to run against an unlisted or failed-gate model, it only
        # warns, so a gate failure below doesn't block the score run either.
        # Decided by the same helper the preview uses -- see _gate_first_argv.
        gate_first_argv = self._gate_first_argv(args)

        self._run_worker(model, argv, log, progress, gate_first_argv)

    @work(exclusive=True)
    async def _run_worker(self, model, argv, log: RichLog, progress: ProgressBar,
                           gate_first_argv=None) -> None:
        if gate_first_argv:
            log.write("[dim]Model not catalogued, so gate-checking first "
                      "(untick \"Gate-check the model first\" to skip this):[/dim]")
            log.write(f"[dim]$ {' '.join(gate_first_argv)}[/dim]")
            gate_result = await run_job(gate_first_argv, on_line=lambda line: log.write(line),
                                        live_log=True)
            log.write(_saved_log_line(gate_result))

            from openllm_cbench.core.gate import summarize_gate_output
            from openllm_cbench.core.registry import load_registry, lookup
            summary = summarize_gate_output(gate_result.lines)
            if gate_result.error:
                log.write(f"[bold red]Gate-check process itself failed to run: {gate_result.error}; "
                          f"nothing was learned about this model automatically.[/bold red]")
            elif summary["hard_failure"]:
                log.write(
                    "[bold red]Gate-check could not reach the endpoint for this model "
                    f"({summary['reason']}): the score run below makes the identical call and "
                    "will most likely fail the exact same way. Check the endpoint/model tag before "
                    "waiting on it.[/bold red]"
                )
            elif summary["clean"] is False and summary["caveats"]:
                log.write(
                    f"[bold yellow]Gate-check ran and found {len(summary['caveats'])} "
                    "caveat(s). Proceeding to score anyway (this framework never blocks a run "
                    "on gate status), but read these first:[/bold yellow]"
                )
                for c in summary["caveats"]:
                    log.write(f"[bold yellow]  - {c}[/bold yellow]")
            elif summary["clean"]:
                log.write("[green]Gate-check clean: no caveats.[/green]")
            else:
                log.write("[bold yellow]Gate-check exited non-zero but printed no recognizable "
                          "report. Proceeding to score anyway; see the full log above for "
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
        rows_total = None

        def on_line(line):
            nonlocal trial_headers_seen, rows_total
            # Rows once the run has said how many there will be (the
            # `Plan:` line, from each suite's own --plan); whole trials
            # only when it could not. See core/plan.py for why trials are
            # the wrong unit for an estimate.
            total = parse_total(line)
            if total:
                rows_total = total
                progress.update(total=total, progress=0)
            elif rows_total is not None:
                if is_row_line(line) and (progress.progress or 0) < rows_total:
                    progress.advance(1)
            elif parse_trial_header(line) is not None:
                if trial_headers_seen > 0:
                    progress.advance(1)
                trial_headers_seen += 1
            if should_show(line):
                log.write(line)

        result = await run_job(argv, on_line=on_line, live_log=True)
        if progress.display and progress.total:
            progress.update(progress=progress.total)
        _report_job_result(log, result,
                            "Check \"Local models\" for the updated Score column, or "
                            "\"Browse reports\" for the full scorecard.")


_ABOUT_TEXT = """\
[bold]About this framework[/bold]

openllm-cbench was built collaboratively with an AI coding assistant \
(Claude Code), not as a demo of that, but because the discipline an \
assistant like that is good at (reading its own prior output critically, \
writing a regression test for every real bug before moving on, checking \
a claim against the actual code instead of memory) turned out to matter \
a lot for a tool whose whole job is measuring whether a model's stated \
capabilities match what it actually does.

[bold]Extending it yourself[/bold]

This project deliberately doesn't assume you use any one AI tool. \
That's why it has a CONTRIBUTING.md instead of a tool-specific config \
file. If you want to extend a suite, add a model to the catalogue, or \
build a new scoring metric with an AI coding assistant's help, any of \
these work the same way: open this repo in [bold]Claude Code[/bold] or \
[bold]Claude Cowork[/bold], or in [bold]Codex CLI[/bold] / \
[bold]ChatGPT Cowork[/bold], and point it at:

  - ARCHITECTURE.md:     what every suite measures and why, the control
                         inventory, and "how not to fool yourself with
                         this tool" (the most transferable section)
  - CONTRIBUTING.md:     layout, conventions, and the standing rules
                         this project has learned the hard way (gate-check
                         both think states, never branch suite code on a
                         model name, smoke-test before calling something
                         done)
  - tests/:              the parity and safety-invariant tests any
                         change should keep passing

None of this requires a specific vendor. The instructions above are \
written to make sense to a human reading them cold, and that's also \
what makes them make sense to any AI assistant you point at them.
"""


class SettingsScreen(Screen):
    """Persistent settings. Today that is one thing: where results go.

    Writes through `cbench config --set-results-dir`, the same subcommand
    a terminal user runs, because every action this app takes is a real
    cbench invocation rather than a second implementation of one."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield InvariantBar()
        with Vertical(id="settings-form"):
            yield Static("[bold]Where results are kept[/bold]", id="settings-title")
            yield Static("", id="settings-current")
            yield Static(
                "Results used to resolve to `./results` next to whatever directory you "
                "launched from, so running this app from one place and the CLI from "
                "another produced two unrelated results trees with the same name. That "
                "is worse than untidy: `cbench score` reads whichever tree it is pointed "
                "at, so a run in the other one is silently missing from the aggregate.\n\n"
                "Setting a location here stores it for your user and applies everywhere, "
                "for both this app and the command line. An explicit --results-dir flag "
                "or $OPENLLM_CBENCH_RESULTS_DIR still wins over it for a single run.",
                id="settings-help",
            )
            yield Input(placeholder="Full path, e.g. C:\\Users\\you\\cbench-results",
                        id="settings-results-input")
            yield Static(
                "[bold]Model catalogue (models.json)[/bold]. Kept separate on "
                "purpose: one catalogue can serve several results folders. An "
                "unpinned catalogue does not lose anything, it silently presents a "
                "DIFFERENT one, so models you have already gate-checked come back as "
                "uncatalogued and the next run goes out ungated.",
                id="settings-models-label",
            )
            yield Input(placeholder="Catalogue file path",
                        id="settings-models-input")
            with Horizontal(id="settings-buttons"):
                yield Button("Save results location", id="settings-save", variant="primary")
                yield Button("Save catalogue location", id="settings-save-models")
                yield Button("Clear results location (back to ./results)", id="settings-unset")
                yield Button("Find existing results", id="settings-find")
                yield Button("Back", id="settings-back")
            yield RichLog(id="settings-log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_current()

    def _refresh_current(self) -> None:
        from openllm_cbench.core.config import config_path, resolution

        try:
            widget = self.query_one("#settings-current", Static)
            field = self.query_one("#settings-results-input", Input)
        except NoMatches:
            return
        from openllm_cbench.core.config import models_resolution

        path, source, pinned = resolution()
        m_path, m_source, m_pinned = models_resolution()
        colour = "green" if pinned else "yellow"
        m_colour = "green" if m_pinned else "yellow"
        widget.update(
            f"Currently: [bold {colour}]{path}[/bold {colour}]\n"
            f"Decided by: {source}\n"
            f"Catalogue: [bold {m_colour}]{m_path}[/bold {m_colour}]\n"
            f"           [dim]{m_source}[/dim]\n"
            f"Config file: {config_path()}"
            f"{'' if config_path().is_file() else '  (does not exist yet)'}"
        )
        if not field.value:
            field.value = str(path)
        try:
            m_field = self.query_one("#settings-models-input", Input)
        except NoMatches:
            return
        if not m_field.value:
            m_field.value = str(m_path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        log = self.query_one("#settings-log", RichLog)
        if event.button.id == "settings-back":
            self.app.pop_screen()
        elif event.button.id == "settings-save":
            path = self.query_one("#settings-results-input", Input).value.strip()
            if not path:
                log.write("[bold red]Enter a path first.[/bold red]")
                return
            self._run(["--set-results-dir", path], log)
        elif event.button.id == "settings-save-models":
            path = self.query_one("#settings-models-input", Input).value.strip()
            if not path:
                log.write("[bold red]Enter a catalogue path first.[/bold red]")
                return
            self._run(["--set-models-file", path], log)
        elif event.button.id == "settings-unset":
            self._run(["--unset-results-dir"], log)
        elif event.button.id == "settings-find":
            self._run(["--find-results"], log)

    def _run(self, args, log) -> None:
        argv = cbench_command("config", args)
        log.write(f"[dim]$ {' '.join(argv)}[/dim]")
        self._config_worker(argv, log)

    @work(exclusive=True, group="settings")
    async def _config_worker(self, argv, log) -> None:
        result = await run_job(argv, lambda line: log.write(line))
        if result.returncode == 0:
            # The location may have just changed, so re-read it rather
            # than leaving the screen showing what it used to be.
            self._refresh_current()
        else:
            log.write(f"[bold red]{exitcodes.describe(result.returncode, 'config')}[/bold red]")


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
                f"Reports in [bold]{root}[/bold] (trial summaries and scorecards). "
                f"Pick a row to read it. Newest first. Raw CSVs and run logs are in the "
                f"tree below.",
                id="reports-caption",
            )
            with Horizontal(id="reports-filter-row"):
                # Scorecards first and alone by default: that is the
                # document almost everyone opens this screen to read, and
                # a model with six trials produces twenty-odd single-run
                # reports that bury it.
                yield Select(
                    [("Scorecards", "scorecard"),
                     ("Trial summaries", "trial summary"),
                     ("Single runs", "single"),
                     ("Everything", "all")],
                    id="reports-kind-select", value="scorecard", allow_blank=False,
                )
                yield Select([("All models", "*")], id="reports-model-select",
                             value="*", allow_blank=False)
            with Horizontal(id="reports-body"):
                with Vertical(id="reports-left"):
                    yield DataTable(id="reports-table")
                    if root.exists():
                        yield DirectoryTree(str(root), id="reports-tree")
                    else:
                        yield Static(
                            f"No results directory at {root}.\n\n"
                            f"If you have run suites before, they wrote somewhere else. With "
                            f"no location pinned, results/ is relative to where you launch "
                            f"cbench: pin one under Settings so every launch finds the same "
                            f"folder, or set $OPENLLM_CBENCH_RESULTS_DIR.",
                            id="reports-empty",
                        )
                with VerticalScroll(id="reports-viewer-container"):
                    yield TextArea("", id="reports-viewer", read_only=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#reports-table", DataTable)
        # Explicit widths. Auto-sizing in a split pane squeezed Date down
        # to "2026", which is not a date -- a column that cannot show its
        # value is worse than one that is not there, because it looks like
        # data.
        table.add_column("Model", width=30)
        table.add_column("Suite", width=15)
        table.add_column("Kind", width=15)
        table.add_column("Date", width=17)
        table.add_column("File")
        table.cursor_type = "row"
        self._paths = {}
        self._found = []
        self._load_reports()

    # Sort order when nothing is filtered out: the document people
    # actually cite first, then the pooled summaries, then the individual
    # runs, each newest-first inside its group.
    KIND_ORDER = {"scorecard": 0, "trial summary": 1}

    def _kind_rank(self, kind):
        return self.KIND_ORDER.get(kind, 2)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in ("reports-kind-select", "reports-model-select"):
            self._render_rows()

    def _render_rows(self) -> None:
        """Re-renders from the already-scanned list. Filtering must not
        re-walk the results tree: that is hundreds of stat() calls to
        answer a question already answered."""
        try:
            table = self.query_one("#reports-table", DataTable)
            kind_sel = self.query_one("#reports-kind-select", Select).value
            model_sel = self.query_one("#reports-model-select", Select).value
        except NoMatches:
            return

        def keep(row):
            _, _, _, _, model, kind = row
            if model_sel not in ("*", Select.BLANK) and model != model_sel:
                return False
            if kind_sel in ("all", Select.BLANK):
                return True
            if kind_sel == "single":
                return kind.endswith("single run")
            return kind == kind_sel

        rows = [r for r in self._found if keep(r)]
        rows.sort(key=lambda r: (self._kind_rank(r[5]), -r[1]))

        table.clear()
        self._paths = {}
        for pth, _sort, when, model, _m2, kind in [
                (r[0], r[1], r[3], r[4], r[4], r[5]) for r in rows]:
            suite = self._classify(pth, _results_root())[1]
            key = table.add_row(model, suite, kind, when, pth.name)
            self._paths[key] = pth

        try:
            caption = self.query_one("#reports-caption", Static)
        except NoMatches:
            return
        shown = len(rows)
        total = len(self._found)
        scope = {"scorecard": "scorecard(s)", "trial summary": "trial summary/summaries",
                 "single": "single-run report(s)", "all": "report(s)"}.get(kind_sel, "report(s)")
        extra = "" if shown == total else f" of {total} total"
        table_empty = ("  [dim]Nothing matches this filter: switch to "
                       "\"Everything\" to see what is there.[/dim]" if shown == 0 else "")
        caption.update(
            f"[bold]{shown}[/bold] {scope}{extra} in [bold]{_results_root()}[/bold]. "
            f"Pick a row to read it. A date marked [b]~[/b] came from the file's mtime "
            f"rather than its name, so it is when the file was last touched, not when the "
            f"run happened. Raw CSVs and run logs are in the tree below.{table_empty}"
        )

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
                    # Prefer the timestamp the suite wrote INTO the
                    # filename over the one the filesystem happens to be
                    # carrying. mtime belongs to whatever last touched
                    # the file, so a results tree that has been through
                    # git, a zip, or a copy shows every report as having
                    # been written at the moment it was unpacked. See
                    # core/runclock.py.
                    stamped = time_from_filename(pth.name)
                    mtime = pth.stat().st_mtime
                    sort_key = stamped.timestamp() if stamped else mtime
                    rows.append((pth, sort_key, stamped, mtime))
            return sorted(rows, key=lambda r: r[1], reverse=True)

        found = await asyncio.to_thread(scan)
        try:
            self.query_one("#reports-table", DataTable)
        except NoMatches:
            return

        root_now = _results_root()
        rows = []
        models = set()
        for pth, sort_key, stamped, mtime in found:
            model, _suite, kind = self._classify(pth, root_now)
            if stamped:
                when = stamped.strftime("%Y-%m-%d %H:%M")
            else:
                # No stamp in the name, so this is the filesystem's
                # opinion and is marked as such rather than shown as if
                # it were the run time.
                when = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M~")
            rows.append((pth, sort_key, stamped, when, model, kind))
            models.add(model)
        self._found = rows

        # Offer only models that actually have a report here. A picker
        # listing tags with nothing behind them is a way to select an
        # empty table.
        try:
            picker = self.query_one("#reports-model-select", Select)
        except NoMatches:
            picker = None
        if picker is not None:
            current = picker.value
            options = [("All models", "*")] + [(m, m) for m in sorted(models)]
            picker.set_options(options)
            picker.value = current if current in {v for _, v in options} else "*"

        self._render_rows()

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
    #run-form, #gate-form, #models-body, #pull-form, #score-form { padding: 1; }
    RichLog { height: 1fr; border: solid $accent; }
    #reports-body { height: 1fr; }
    #reports-tree { width: 40%; }
    #reports-viewer-container { width: 60%; }
    #models-buttons { height: auto; }
    #models-buttons Button { margin: 0 1 0 0; }
    #settings-form { padding: 1 2; }
    #settings-help { margin: 1 0; }
    #settings-buttons { height: auto; margin: 1 0; }
    #settings-buttons Button { margin: 0 1 0 0; }
    #settings-log { height: 1fr; border: round $panel; }
    #results-location { margin: 0 0 1 0; }
    #reports-filter-row { height: auto; margin: 0 0 1 0; }
    #reports-filter-row Select { width: 1fr; margin: 0 1 0 0; }
    #models-delete-row { height: auto; margin: 0 0 1 0; }
    #models-delete-row Checkbox { width: 1fr; }
    #models-limit-row { height: auto; margin: 0 0 1 0; }
    #models-limit-label { width: auto; padding: 1 1 0 0; }
    #models-limit-suffix { width: auto; padding: 1 0 0 1; }
    #models-limit-input { width: 12; }
    #models-table { height: 12; border: solid $accent; }
    #score-form { height: 1fr; padding: 0 1; }
    .score-heading { margin: 1 0 0 0; }
    #score-suite-checks { height: auto; }
    #score-suite-checks Checkbox { margin: 0 2 0 0; }
    #score-budget-row { height: auto; }
    #score-budget-row Input { width: 1fr; }
    #score-run { height: auto; padding: 0 1; }
    #score-buttons { height: auto; }
    #score-buttons Button { margin: 0 1 0 0; }
    #score-log { height: 10; }
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
