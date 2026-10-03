# Setup

cbench tests a model that you serve yourself. Before the first real run you
need three things: Ollama serving the model, cbench installed, and two
locations pinned so that every run reads and writes the same files. This
page covers them in that order.

## Ollama

### Install Ollama and pull a model

cbench does not run models itself. It sends every model request to a model
server, by default Ollama on this machine. Install Ollama from
[ollama.com/download](https://ollama.com/download):

- **Windows and macOS:** the Ollama app starts the server and keeps it
  running in the background.
- **Linux:** the install script sets the server up as a systemd service
  named `ollama`. Without the service, start the server with `ollama serve`
  and leave that terminal open.

Then pull a model and check that the server lists it:

```bash
ollama pull <model-tag>   # for example qwen3:4b
ollama list
```

Any model that `ollama list` shows can be tested, and `cbench gate` says
which suites can measure it before you spend a run on it. Without leaving
cbench, `cbench search` checks that a tag exists without downloading it and
`cbench pull` downloads it; see
[Populating the catalogue](MODEL_CATALOGUE.md#populating-it).

### The endpoint

cbench talks to Ollama's native API at `http://localhost:11434` by default.
To use another endpoint, set `$OPENLLM_CBENCH_ENDPOINT`, which every command
that talks to the endpoint reads. `--endpoint` overrides it on `doctor`,
`gate`, `discover`, `catalogue`, `search`, `pull`, `remove` and the three
suites; `score`, `assess` and `guardrail` take no `--endpoint` and read the
environment variable only.

The endpoint must implement Ollama's native routes. A server that offers
only an OpenAI-compatible API is not enough: cbench sets the context window
on every request and reads the reasoning trace from its own field, and it
does both through the native API.

| Route | What cbench uses it for |
|---|---|
| `/api/chat` | Every model call: the suites, gate checks and `cbench guardrail` |
| `/api/show` | Model information: capabilities, parameters and quantisation |
| `/api/tags` | The list of pulled models |
| `/api/ps` | GPU residency, read after every successful model call in a suite |
| `/api/version` | The version `cbench doctor` reports |
| `/api/pull`, `/api/delete` | `cbench search`, `cbench pull` and `cbench remove` |

### Keep Ollama on loopback

Ollama has no authentication. By default it listens only on loopback
(`127.0.0.1:11434`), which only this machine can reach. Setting
`OLLAMA_HOST=0.0.0.0` (or a LAN address), or turning on "Expose Ollama to
the network" in the Ollama app, lets anyone who can reach the port use your
models and GPU, and delete models.
[Securing the model endpoint](../SECURITY.md#securing-the-model-endpoint)
covers what is exposed and what to do if you need remote access.

For an endpoint on this machine, `cbench doctor` reads the listening
address from the running server's startup log, which the Ollama app writes
on Windows and macOS. When there is no log to read, it reports nothing
about the network setting: on Linux the service logs to the systemd
journal, and a server started by hand with `ollama serve` logs to its own
terminal. Check it yourself:

```bash
ss -ltn | grep 11434   # the default port; 127.0.0.1:11434 means loopback only
systemctl cat ollama   # the service's unit file and overrides, including any OLLAMA_HOST
```

On macOS, neither command exists; see [Ollama on macOS](#ollama-on-macos).

### Model storage

Each model takes several gigabytes of disk. Ollama keeps them in
`~/.ollama/models` by default (`/usr/share/ollama/.ollama/models` for the
Linux service), or wherever `OLLAMA_MODELS` or the Ollama app's model
location setting points. `cbench doctor` reports the folder in use and
checks it against the models the server lists.

### GPU memory

A model runs fastest when all of it fits in GPU memory. A model that does
not fit still answers: Ollama runs the rest on the CPU and the model
generates the same text more slowly, so more requests hit their timeout
before they finish, and those rows leave the rates. Every suite records how much of the model was in GPU memory (see
[GPU residency is recorded](USER_GUIDE.md#gpu-residency-is-recorded-because-a-model-that-does-not-fit-is-silent-too)),
and `cbench doctor` reports the hardware headroom.

Other work on the same GPU (another model loaded in Ollama, another model
server, a game) takes memory and time from a run. cbench refuses to start a
second assessment while one is running, but it cannot see other programs,
so close them before a long run.

On Apple Silicon the GPU shares the machine's memory, up to a cap that
macOS sets; see [GPU memory on Apple Silicon](#gpu-memory-on-apple-silicon).

### Context length

The context length setting in the Ollama app, and `OLLAMA_CONTEXT_LENGTH`,
do not change a cbench run. Every chat request cbench sends carries its own
context window (`num_ctx`) and reply limit (`num_predict`), and the suites
size both to the model; see
[Generation budgets](USER_GUIDE.md#generation-budgets).

### Server settings cbench does not record

cbench records the sampling parameters, the generation budget and GPU
residency in every row, but not settings applied to the whole server. Some
of those change how a model is computed or how much memory it needs, such
as `OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE` and
`OLLAMA_NUM_PARALLEL` (which multiplies the memory set aside for the context
window, and so affects GPU residency). Keep them unchanged
across runs you intend to pool or compare, and state them when you publish
a grade.

### Ollama versions

cbench does not check the Ollama version; `cbench doctor` reports it.
cbench has been run against Ollama 0.34.2, 0.34.4 and 0.35.0 (macOS). If a gate check fails
on a much older server, update Ollama before treating the failure as a
property of the model.

## Install cbench

cbench needs Python 3.10 or later. The package is not on PyPI, so install
it from GitHub.

To install it with the terminal UI:

```bash
pip install "openllm-cbench[tui] @ git+https://github.com/gfitzp79/llm-cbench"
cbench doctor
```

Leave out `[tui]` for the CLI alone; its only dependency is `requests`. If
your system Python refuses `pip install` with "externally-managed-environment"
(for example on recent Ubuntu releases, and Homebrew's Python on macOS), use
a virtual environment, or `pipx install` with the same argument.

To read or change the code, run the tests, or use the Inspect
cross-validation (its task files run by path, so it needs a clone):

```bash
git clone https://github.com/gfitzp79/llm-cbench
cd llm-cbench
pip install -e ".[dev,tui]"      # or ".[dev,tui,inspect]" for Inspect
pytest -q                        # needs no model and no network
```

On macOS, read [macOS](#macos) first: the Python that ships with it is too
old, and the commands above need adjusting.

## macOS

Measured on macOS 27 on an Apple M5 with 32 GB of memory, Ollama 0.35.0
and Python 3.12 from Homebrew. Everything in this guide applies; this
section covers what differs.

### Install cbench on macOS

The `python3` that ships with macOS (`/usr/bin/python3`) is 3.9, too old
for cbench, and there is no `pip` command, so the
[Install cbench](#install-cbench) commands fail on a Mac as written: with
`command not found: pip`, or, through `pip3`, with
`No matching distribution found for requests>=2.33.0`, which reads like a
network problem and is not one. Homebrew's Python refuses `pip install`
outside a virtual environment (`externally-managed-environment`). So
install a supported Python, then cbench with `pipx`, which gives it its own
environment and puts the `cbench` command on your `PATH` in every terminal.

**1. Homebrew.** Run `brew --version`. If that prints a version, go to
step 2. Otherwise install Homebrew with the command on
[brew.sh](https://brew.sh); it also installs the Xcode Command Line Tools,
which provide `git`. When it finishes, it prints "Next steps": run the
commands it shows, which put `brew` on your `PATH`, and check that
`brew --version` now works.

**2. Python and pipx.**

```bash
brew install python@3.12 pipx
```

Name the Python version. Homebrew's plain `python` formula is 3.14,
outside the 3.10 to 3.13 range that CI tests on macOS. pipx pulls in
Homebrew's newest Python for itself; that is expected, and step 4 keeps
cbench on 3.12.

**3. Put pipx's commands on your PATH.**

```bash
pipx ensurepath
```

**Then close the terminal and open a new one.** `pipx ensurepath` adds
`~/.local/bin` to `PATH` in `~/.zshrc`, which only a new terminal reads.
Skip this and the next steps still install cbench, but `cbench` answers
`zsh: command not found: cbench` in this terminal.

**4. Install cbench**, in the new terminal:

```bash
pipx install --python python3.12 "openllm-cbench[tui] @ git+https://github.com/gfitzp79/llm-cbench"
```

Keep the quotes: zsh, the macOS default shell, reads an unquoted `[tui]`
as a filename pattern.

**5. Check it.**

```bash
cbench --version
cbench doctor
```

Then continue with [Where results are kept](#where-results-are-kept), and
`cbench tui` for the terminal UI.

**To update** to the latest version on GitHub, run
`pipx reinstall openllm-cbench`. `pipx upgrade` does not fetch a newer
GitHub version: it checks only a package index, and cbench is not on one.

#### From a clone

To change the code or run the tests, clone the repository and install it
editable, so that `git pull` updates the `cbench` command without a
reinstall (steps 1 to 3 above still apply):

```bash
git clone https://github.com/gfitzp79/llm-cbench
cd llm-cbench
pipx install --python python3.12 --editable ".[tui]"
```

Run the tests from a virtual environment inside the clone:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,tui]"
.venv/bin/pytest -q
```

That environment's own `cbench` is on `PATH` only in a terminal where it
has been activated (`source .venv/bin/activate`), which is the usual
cause of `zsh: command not found: cbench` in a new terminal; the `pipx`
install above is the one to use day to day. The TUI tests print a few
`Event loop is closed` warnings on Python 3.12; they are harmless.

pipx keeps cbench's environment in `~/Library/Application Support/pipx`
and links the command into `~/.local/bin`. To remove it:
`pipx uninstall openllm-cbench`. Your results and settings are kept
elsewhere (see the table below), so they survive a reinstall.

### Ollama on macOS

The Ollama app runs the server and keeps it running. It takes its
settings from its own Settings window and from `launchd`, not from your
shell, so `export OLLAMA_HOST=...` in a terminal changes nothing the
server reads.

Check what the server listens on (the Linux commands under
[Keep Ollama on loopback](#keep-ollama-on-loopback) do not exist on macOS):

```bash
lsof -nP -iTCP:11434 -sTCP:LISTEN   # 127.0.0.1:11434 is loopback only; *:11434 is every interface
launchctl getenv OLLAMA_HOST        # a value set for the app through launchd, if any
```

`*:11434` means "Expose Ollama to the network" is on in the app's
settings, or `OLLAMA_HOST` was set with `launchctl setenv`. Turn the
setting off, or run `launchctl unsetenv OLLAMA_HOST`, then quit and reopen
the app. `cbench doctor` reads the address from the server's startup log
and warns when it is not loopback.

| What | Where on macOS |
|---|---|
| Ollama executable | `/Applications/Ollama.app/Contents/Resources/ollama`, linked as `/usr/local/bin/ollama` |
| Server logs | `~/.ollama/logs`; `server.log` is the running session, older ones rotate to `server-1.log` and on |
| Models | `~/.ollama/models`, unless the app's settings point elsewhere |
| cbench config file | `~/.config/openllm-cbench/config.json` (not `~/Library/Application Support`) |
| cbench run lock | `~/.cbench/run.lock` |

### GPU memory on Apple Silicon

Apple Silicon has no separate video memory: the GPU uses part of the
unified memory, and macOS caps how much. On the 32 GB machine above, the
cap was 25 GB, which Ollama logs at startup
(`msg="inference compute" ... library=Metal ... total="25.0 GiB"`). A
model larger than the cap still runs, partly on the CPU, and every suite
records that as GPU residency below 100%, as it does on any GPU.

`cbench doctor`, the TUI's Fit column and every scorecard use that cap,
not total memory, as this machine's GPU memory. It comes from the first of
these that is available: your own `iogpu.wired_limit_mb` setting, the
figure in Ollama's log, or an estimate of 75% of memory, which doctor
labels as an estimate. To give the GPU more, at the cost of memory for
macOS and everything else:

```bash
sudo sysctl iogpu.wired_limit_mb=28000   # in MB; lasts until restart
```

Quit and reopen Ollama afterwards, since it reads the cap when it starts.
A grade records the GPU memory it was produced with, so runs at different
caps are runs on different hardware (see
[Responsible use](../README.md#responsible-use)).

On an Intel Mac, Ollama runs models on the CPU, and `cbench doctor` reports
no GPU.

### Long runs

A run takes longer than the model's speed suggests, because every suite
sends many requests: `cbench score --depth quick` on a 0.6B model took 16
minutes on the M5 above, and the default depth runs three times as many
trials. A Mac that sleeps mid-run stalls it. Keep it awake for the length
of the command:

```bash
caffeinate -i cbench score --model <model-tag>
```

`caffeinate -i` prevents idle sleep only; closing a MacBook's lid still
sleeps it.

The canary listens on `127.0.0.1` only, so the macOS firewall does not
prompt for it.

## Where results are kept

**Set this once, before your first real run:**

```bash
cbench config --set-results-dir ~/cbench-results
```

Without it, results go to `./results` under whichever directory you launch
from. Launch the TUI from your home directory and the CLI from a project,
and you get two unrelated results folders with the same name, each
invisible to the other. That is worse than untidy: `cbench score` reads
only the folder it resolves to, so a run that landed in the other one is
missing from the aggregate without any warning, and the grade is computed
over whatever subset shared a directory with it.

`cbench config` on its own prints where results are going and which setting
decided that. `cbench config --find-results` searches the usual places for
results folders that already exist; run it once if you have used the tool
from more than one directory. It only reads, and moves nothing.

`cbench doctor` lists every location at once, each with the setting or
check behind it: Ollama's executable, logs and model storage, and cbench's
config file, results folder, catalogue, TUI logs and run lock. Ollama's
model folder is read from the running server's own startup log and
confirmed by matching its manifests against the models the server lists,
because the platform default can exist, be empty and be wrong: a folder
chosen in the Ollama app's settings is not an environment variable any
other process can see. For an endpoint on this machine, `cbench doctor`
also warns when the server listens beyond loopback (`OLLAMA_HOST` set to
`0.0.0.0` or a LAN address), because Ollama has no authentication; see
[Keep Ollama on loopback](#keep-ollama-on-loopback) for when it can check.

Resolution order, most specific first:

| Setting | Scope |
|---|---|
| `--results-dir` on a command that takes it (the three suites and `compare`) | that invocation |
| `$OPENLLM_CBENCH_RESULTS_DIR` | that shell |
| the config file | your user, in every directory |
| `./results` | whichever directory you are in |

The config file is `%APPDATA%\openllm-cbench\config.json` on Windows and
`$XDG_CONFIG_HOME/openllm-cbench/config.json` (by default
`~/.config/openllm-cbench/config.json`) elsewhere; `$OPENLLM_CBENCH_CONFIG`
overrides its location. The same settings are available in the TUI under
**Settings**, and the dashboard shows the active results location on every
launch, in yellow when it is not pinned.

**Pin the model catalogue too**, for the same reason and with the same four
layers (`--registry-file` on a command that takes it,
`$OPENLLM_CBENCH_MODELS_FILE`, the config file, then `./models.json`):

```bash
cbench config --set-models-file ~/cbench-results/models.json
```

An unpinned catalogue is worse than an unpinned results folder, because it
loses nothing visibly: it presents a *different* `models.json` without
saying so. Models you have already gate-checked come back as uncatalogued,
and the next run goes out ungated, without the configuration those gate
checks recorded. The catalogue is a separate setting rather than derived
from the results location because one catalogue can serve several sets of
results, and moving your results should not mean re-gating every model.
`--unset-results-dir` and `--unset-models-file` remove a pinned location.

## What goes in `--model <model-tag>`

Commands that act on one model take `--model <model-tag>`. The tag is
whatever your endpoint calls the model, passed through unchanged in every
request this framework sends. For the default (Ollama) endpoint it is the
`NAME` column of `ollama list`:

```bash
$ ollama list
NAME                    ID              SIZE      MODIFIED
gemma3:12b              f4031aab637d    8.1 GB    8 weeks ago
qwen3.5:9b              6488c96fa5fa    6.6 GB    12 days ago
hf.co/org/repo:Q4_K_M   ...

$ cbench doctor              # takes no --model: checks the endpoint only
$ cbench gate --model gemma3:12b --save
```

There is no separate registration step and no fixed roster: if your
endpoint recognises the tag, `cbench` can run a suite against it. A wrong
tag fails at the first command that uses it, with a message that says the
model could not be reached: `cbench gate` exits `2`, and the pre-flight in
`cbench score` and `cbench assess` refuses to start (also exit `2`). It does
not surface later as a confusing result inside a suite's output.
`cbench doctor` checks only the endpoint, so it cannot catch a wrong tag.
