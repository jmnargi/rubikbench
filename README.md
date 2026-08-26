# RubikBench

RubikBench tests how well a large language model (LLM) can solve a 3x3x3 Rubik's cube.

The model uses tools to apply moves to the cube; the current cube state is
always provided with every message. The benchmark runs on its own. It connects
to any OpenAI-compatible endpoint.

RubikBench has a Textual TUI. The TUI has three screens:

- Configuration.
- Live run.
- Results.

## Terms

| Term | Meaning |
|---|---|
| Conversation turn | One model reply in the benchmark loop. |
| Endpoint | A server URL that accepts chat-completion requests. |
| Extra body parameters | JSON values sent with every chat-completion request. |
| Max input tokens | The maximum number of tokens in one request. RubikBench trims the older conversation turns to keep the request below this value. |
| Max output tokens | The maximum number of tokens in one model reply. RubikBench sends this value as "max_tokens" in the request body. |
| Move | One face turn of the cube, for example "R" or "U'". |
| Par | The reference number of moves to solve the scramble. |
| Scramble | The starting mixed state of the cube. |
| Starting scramble set | The source of the scrambles. It is random, a premade set, or a custom file. |
| Tool call | A request from the model to execute one tool. |
| Turn budget | The maximum number of conversation turns for one solve. |

## 1. Install

**Requirement:** Python 3.10 or newer.

Run this command:

```bash
uv sync --extra solver --extra dev
```

The solver extra installs kociemba. Kociemba gives the par value for scoring. It is optional. Without kociemba, RubikBench uses the value 20 for par.

## 2. Run the benchmark

Run this command to start a benchmark immediately (it reads `.env`, uses
CLI flags if given, and streams the model's output live):

```bash
uv run rubikbench
```

`uv run rubikbench run` is identical. There is no config menu and nothing to
click: if `.env` is missing something and no flag provides it, the command
prints what's missing and exits with a pointer to `rubikbench --help`.

For a full-screen live view, run:

```bash
uv run rubikbench tui
```

The TUI also starts the benchmark from `.env` immediately. It is view-only:
live streaming model output, tool calls, cube state, and realtime stats.
Press Ctrl+C (or `q`) to quit.

## 3. Where settings come from

Precedence: CLI flags > `.env` > `rubikbench_config.json` (if present) >
built-in defaults. See section 8 for the full `.env` and flag reference.

## 4. Run the benchmark

The live run screen shows:

- The current cube, with colors.
- The move history.
- The statistics. The statistics are turns, tool calls, moves, elapsed time, and score.
- The log of the model activity, streaming live as the model replies.

Press Ctrl+C (or `q`) to quit at any time.

## 5. How a solve works

For each solve, the benchmark does these steps:

1. Get the scramble from the starting scramble set. The default set is random. The default length is 22 moves. The generator does not repeat the same face two times in a row.
2. Give the cube state to the model. The model does not receive the scramble. It must solve the cube from the state alone.
3. Ask the model for a reply. Each reply is one conversation turn.
4. Execute the tool calls in the reply.
5. Check the cube. If it is solved, the solve is complete.
6. Repeat steps 3 to 5 until one of these conditions is true:
   - The cube is solved.
   - The turn budget is empty.
   - An API error stops the run.
   - You abort the run.

The model has two tools:

- `apply_moves` — applies one or more moves in one call. The moves use Singmaster notation, for example "R U R' U'". Invalid tokens are not applied.
- `reset_cube` — returns the cube to the original scramble. It costs conversation turns. It does not add moves.

The current cube state is always provided: it is in the initial message, and
every tool result includes the updated state. There is no `get_cube_state`
tool, so the model never wastes a call on observation. Reasoning models'
chain-of-thought (when the provider sends it) streams live alongside the
answer and is recorded in the transcript.

The model can apply many moves in one tool call. It can also make more than one tool call in one reply.

### Moves as text

Some models write moves as text instead of using the tool. RubikBench can parse these moves and apply them. The configuration value "allow text moves" controls this behavior. The default is on.

### Retries

If a request fails, RubikBench retries it. The number of retries is a configuration value. The default is 2.

RubikBench records the number of retries in the results.

### Truncation

The server can stop an answer before it is complete. The server reports the finish reason "length" in this case. RubikBench records this finish reason. A solve with this finish reason is truncated. The results and the export show the truncated solves.

### Starting scrambles

Every solve starts from a solved cube. RubikBench applies a scramble to the cube.

A scramble from a solved cube is always solvable. The reason: a scramble is a sequence of legal face turns. The inverse of that sequence solves the cube again.

The configuration value "starting scramble set" selects the source of the scrambles. It has three modes:

- Random. This is the default. RubikBench generates a new scramble for each solve. A fixed seed makes the run reproducible.
- A premade set. RubikBench uses the scrambles in the set. The premade sets are:
  - Superflip. It is one scramble. Every edge piece is flipped in place. This state needs 20 moves to solve.
  - Cube in cube. It is one scramble. It makes a pattern with a small cube in each face.
  - Catalog 10, 16, 22, and 25 moves. Each catalog has four fixed scrambles.
- Custom scrambles file. RubikBench reads the scrambles from a text file. The file has one scramble per line.

RubikBench checks each premade and custom scramble before the run. The check applies the scramble to a solved cube and replays the inverse sequence. A bad scramble stops the run with an error message.

When the set has fewer scrambles than the number of solves, the run cycles through the set. For example, a run with six solves and the Catalog 16 moves set uses the first two scrambles two times each.

## 6. Scoring

An unsolved solve scores zero. A solved solve scores from 0 to 1000.

The score has three factors:

| Factor | Formula | Meaning |
|---|---|---|
| Moves | `min(1, par / moves)` | How close the solution is to the par value. |
| Conversation turns | `(turn_budget - turns + 1) / turn_budget` | How few replies the model needed. |
| Tool calls | `min(1, turns / tool_calls)` | Tool discipline. One tool call per turn is ideal. |

The score is this formula:

`1000 x (0.5 x moves_factor + 0.3 x turns_factor + 0.2 x tools_factor)`

The weights are configuration values. You can change them.

The benchmark records the elapsed time and the token usage for each solve. It does not include them in the score.

## 7. Results

The results screen shows:

- The aggregate statistics. They include the solve rate and the average score.
- A table with one row per solve. Each row shows the turns, tool calls, moves, par, time, and score.
- The details of the selected solve. The details have three tabs:
  - Summary — the score breakdown.
  - Moves — the scramble and the applied moves.
  - Transcript — the complete model activity.

The results screen also shows the token data and the finish reasons. The token data has the input tokens, the output tokens, and the cached tokens for each solve. A finish reason "length" marks a truncated solve.

Select "Replay" to review one solve step by step. The replay screen shows the cube and a timeline. The timeline has one entry for every state change. Use the space bar to play or pause. Use the arrow keys to move one step. Use the plus and minus keys to change the speed. Select another solve to replay it.

Select "Export JSONL" to save the results. The output file is in the directory `rubikbench_results`.

## 8. Run the benchmark without the TUI

### Run from .env (no config file)

For a quick headless run you can skip the TUI and the config file entirely.
RubikBench loads a `.env` file from the project directory at startup, so
credentials never have to live in a config file (`.env` is gitignored).

Create `.env` with your endpoint and key:

```bash
OPENAI_API_KEY=sk-...
RUBIKBENCH_MODEL=gpt-4o
```

Then run:

```bash
uv run rubikbench run
```

(Plain `uv run rubikbench` does exactly the same thing.)

With no config file present, RubikBench defaults to the OpenAI endpoint and
model; everything else is optional. All supported `.env` variables:

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | OpenAI key; used automatically when the endpoint is OpenAI. |
| `OPENROUTER_API_KEY` | OpenRouter key; used automatically when the endpoint is OpenRouter. |
| `RUBIKBENCH_API_KEY` | Generic API key override for any endpoint. |
| `RUBIKBENCH_BASE_URL` | Endpoint URL (any OpenAI-compatible /v1 server). |
| `RUBIKBENCH_MODEL` | Model name. |
| `RUBIKBENCH_MAX_INPUT_TOKENS` | Context cap; older turns are trimmed to fit. |
| `RUBIKBENCH_MAX_OUTPUT_TOKENS` | Sent as `max_tokens` in the request. |
| `RUBIKBENCH_TEMPERATURE` | Sampling temperature (blank = model default). |
| `RUBIKBENCH_TIMEOUT` | Request timeout in seconds. |
| `RUBIKBENCH_MAX_RETRIES` | Retries per request. |
| `RUBIKBENCH_SOLVES` | Number of solves. |
| `RUBIKBENCH_MAX_TURNS` | Turn budget per solve. |
| `RUBIKBENCH_SCRAMBLE_LEN` | Scramble length. |
| `RUBIKBENCH_SEED` | Fixed scramble seed (reproducible runs). |

When a config file exists, environment values take precedence for the base
URL, model, and API key; everything else comes from the file.

The API key is resolved from `RUBIKBENCH_API_KEY` first, then the env var of
the provider whose URL (or host) matches the endpoint (`OPENAI_API_KEY`,
`OPENROUTER_API_KEY`, ...), then `OPENAI_API_KEY` as the default for any
other OpenAI-compatible endpoint such as a custom LiteLLM proxy. Local
servers (`localhost`, `*.local`) need no key. Run
`uv run rubikbench validate` to check the resolved settings and which
variable the key came from, and `uv run rubikbench presets` to list
providers.

Run this command to list the presets:

```bash
uv run rubikbench presets
```

Run this command to check a configuration file:

```bash
uv run rubikbench validate --config cfg.json
```

Run this command to start a benchmark run:

```bash
uv run rubikbench run --config cfg.json -o results.jsonl
```

`rubikbench run` starts immediately and needs no TUI and no config file:
without `--config` it reads `.env` (see above), and every setting can be
passed as a flag, which overrides `.env` and the config file:

```bash
uv run rubikbench run \
  --base-url https://api.openai.com/v1 \
  --api-key sk-... \
  --model gpt-4o \
  --max-input-tokens 32768 \
  --max-output-tokens 4096 \
  -n 3 --max-turns 40 --scramble-len 22 --seed 42
```

Run `uv run rubikbench run --help` for the full flag list. The model's text
and tool calls stream to the terminal live as they happen.

The run command prints a JSON summary to the standard output. The summary has the solve rate, the average score, the average moves, the tokens, the retries, and the number of truncated solves.

The run command writes one line to the output file for each solve. The first line is the header. It contains the configuration and the aggregate statistics. Each solve line contains the analytics and the full transcript. The analytics have the input tokens, the output tokens, the cached tokens, the retries, the finish reasons, and the truncation flag. Each solve line also contains a timeline. The timeline has the cube state after every state change.

Run this command to merge run files into one dataset file:

```bash
uv run rubikbench aggregate run1.jsonl run2.jsonl -o dataset.json
```

The dataset file is one JSON document. It contains all solve records and the totals. The totals have the tokens, the turns, the tool calls, the moves, the retries, and the truncated solves.

Run this command to replay a run file in the browser:

```bash
uv run rubikbench view results.jsonl
```

The command starts a local web server. The server opens the page in your browser. The page shows the cube in 3D. The cube replays the states in the timeline. You can play, pause, and jump to a step. You can change the replay speed.

The command has two optional flags:

- `--no-open`. Use it to start the server without opening the browser.
- `--port N`. Use it to select the port. The default port is 8321.

Press Control+C to stop the server.

## 9. Run the tests

Run this command:

```bash
uv run pytest
```

The test suite has 106 tests. It covers the cube model, the scramble generator, the scoring, the configuration, the tool loop, the token analytics, the dataset aggregation, the web replay, the headless CLI, and the TUI.

Run this command to check the style:

```bash
uv run ruff check
```

## 10. Project layout

```
rubikbench/
  cube.py       # The cube model and the move tables.
  scramble.py   # The scramble generator.
  rendering.py  # The ASCII net and the colored net.
  scoring.py    # The benchmark score.
  config.py     # The configuration values and the presets.
  llm.py        # The OpenAI-compatible client.
  benchmark.py  # The tool execution and the turn loop.
  prompts.py    # The system prompt and the tool schemas.
  solver_ref.py # The kociemba integration.
  aggregate.py  # The dataset aggregation.
  cli.py        # The TUI entry point and the headless commands.
  webui/        # The web replay server and the 3D page.
  tui/          # The Textual application.
```
