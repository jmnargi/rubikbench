# RubikBench

RubikBench measures how well a large language model (LLM) solves a 3x3x3 Rubik's cube.

The model uses tools to make cube moves. Each message includes the current cube state. The benchmark runs without manual steps. It connects to any OpenAI-compatible endpoint.

RubikBench includes a Textual terminal user interface (TUI). The TUI has three screens:

- Configuration
- Live run
- Results

## Terms

| Term | Meaning |
|---|---|
| Conversation turn | One model reply in the benchmark loop. |
| Endpoint | A server URL that accepts chat-completion requests. |
| Extra body parameters | JSON values that RubikBench sends with each chat-completion request. |
| Max input tokens | The maximum number of tokens in one request. RubikBench removes older conversation turns when needed to stay below this limit. |
| Max output tokens | The maximum number of tokens in one model reply. RubikBench sends this value as `max_tokens` in the request body. |
| Move | One face turn of the cube, such as `R` or `U'`. |
| Par | The reference number of moves for a solution. |
| Scramble | The starting mixed state of the cube. |
| Starting scramble set | The source of the scrambles: random, premade, or a custom file. |
| Tool call | A request from the model to execute one tool. |
| Turn budget | The maximum number of conversation turns for one solve. |

## 1. Install

**Requirement:** Python 3.10 or newer.

Run this command:

```bash
uv sync --extra solver --extra dev
```

The `solver` extra installs `kociemba`. RubikBench uses `kociemba` to calculate the par value for scoring. This package is optional. Without it, RubikBench uses 20 as the par value.

## 2. Run the benchmark

Run this command to start a benchmark:

```bash
uv run rubikbench
```

This command reads `.env`, applies any CLI flags, and streams model output to the terminal. `uv run rubikbench run` does the same thing.

The command starts without a configuration menu. If a required value is missing, it prints the missing value and points to `rubikbench --help`.

Run this command for a full-screen live view:

```bash
uv run rubikbench tui
```

The TUI starts the benchmark with the values in `.env`. The TUI is view-only. It shows model output, tool calls, cube state, and live statistics. Press `Ctrl+C` or `q` to quit.

## 3. Settings

RubikBench uses settings in this order:

1. CLI flags
2. `.env`
3. `rubikbench_config.json`, if the file exists
4. Built-in defaults

See [Section 8](#8-run-the-benchmark-without-the-tui) for the complete `.env` and CLI flag reference.

## 4. Live run

The live run screen shows:

- The current cube and its colors
- The move history
- Statistics for turns, tool calls, moves, elapsed time, and score
- A live log of model activity

Press `Ctrl+C` or `q` to quit.

## 5. How a solve works

For each solve, RubikBench performs these steps:

1. It gets a scramble from the starting scramble set. The default set is random. The default scramble length is 22 moves. The generator does not use the same face two times in a row.
2. It sends the cube state to the model. It does not send the scramble. The model must solve the cube from the state.
3. It requests a model reply. Each reply is one conversation turn.
4. It executes the tool calls in the reply.
5. It checks the cube. If the cube is solved, the solve ends.
6. It repeats steps 3 through 5 until one of these conditions is true:
   - The cube is solved.
   - The turn budget is empty.
   - An API error stops the run.
   - You stop the run.

The model has two tools:

- `apply_moves` applies one or more moves in one call. The moves use Singmaster notation, such as `R U R' U'`. RubikBench does not apply invalid tokens.
- `reset_cube` returns the cube to the original scramble. This action uses conversation turns. It does not add moves.

RubikBench always sends the current cube state. It sends the state in the initial message, and each tool result includes the updated state. There is no `get_cube_state` tool. The model does not need a call to read the state. If the provider sends chain-of-thought from a reasoning model, RubikBench streams it with the answer and records it in the transcript.

The model can apply many moves in one tool call. It can also make more than one tool call in one reply.

### Moves as text

Some models write moves as text instead of using the tool. RubikBench can parse and apply these moves. The `allow text moves` setting controls this behavior. The default value is on.

### Retries

If a request fails, RubikBench retries the request. The retry count is a setting. The default value is 2.

RubikBench records the retry count in the results.

### Truncation

A server can stop an answer before the answer is complete. In this case, the server reports the finish reason `length`. RubikBench records this reason. The solve is truncated. The results and export show the truncated solve.

### Starting scrambles

Each solve starts with a solved cube. RubikBench applies a scramble to the cube.

A scramble is always solvable because it is a sequence of legal face turns. The inverse sequence solves the cube.

The `starting scramble set` setting selects the scramble source. It has three modes:

- **Random:** This is the default. RubikBench generates a new scramble for each solve. A fixed seed makes a run reproducible.
- **Premade set:** RubikBench uses scrambles from a built-in set:
  - **Superflip:** One scramble that flips every edge piece in place. It needs 20 moves to solve.
  - **Cube in cube:** One scramble that creates a small cube pattern on each face.
  - **Catalog 10, 16, 22, and 25 moves:** Each catalog has four fixed scrambles.
- **Custom scrambles file:** RubikBench reads one scramble from each line of a text file.

RubikBench checks each premade and custom scramble before the run. It applies the scramble to a solved cube and then applies the inverse sequence. A bad scramble stops the run and displays an error.

If the set has fewer scrambles than the run has solves, RubikBench cycles through the set. For example, a run with six solves and the Catalog 16 moves set uses the first two scrambles two times each.

## 6. Scoring

An unsolved solve has a score of zero. A solved solve has a score from 0 to 1000.

The score uses three factors:

| Factor | Formula | Meaning |
|---|---|---|
| Moves | `min(1, par / moves)` | How close the solution is to the par value. |
| Conversation turns | `(turn_budget - turns + 1) / turn_budget` | How few replies the model needs. |
| Tool calls | `min(1, turns / tool_calls)` | Tool discipline. One tool call per turn is the ideal value. |

The score formula is:

`1000 x (0.5 x moves_factor + 0.3 x turns_factor + 0.2 x tools_factor)`

The weights are settings. You can change them.

RubikBench records elapsed time and token usage for each solve. These values do not affect the score.

## 7. Results

The results screen shows:

- Aggregate statistics, including solve rate and average score
- One table row for each solve, with turns, tool calls, moves, par, time, and score
- Details for the selected solve in three tabs:
  - **Summary:** The score breakdown
  - **Moves:** The scramble and the applied moves
  - **Transcript:** All model activity

The results screen also shows token data and finish reasons. Token data includes input, output, and cached tokens for each solve. The finish reason `length` marks a truncated solve.

Select **Replay** to review a solve step by step. The replay screen shows the cube and a timeline. The timeline has one entry for each state change. Press the space bar to play or pause. Use the arrow keys to move one step. Use the plus and minus keys to change the speed. Select another solve to replay it.

Select **Export JSONL** to save the results. The output file is in the `rubikbench_results` directory.

## 8. Run the benchmark without the TUI

### Run from `.env` without a config file

You can run RubikBench without the TUI or a config file.

RubikBench loads `.env` from the project directory at startup. The file is gitignored, so API keys do not need to be stored in a config file.

Create `.env` with your endpoint and key:

```bash
OPENAI_API_KEY=sk-...
RUBIKBENCH_MODEL=gpt-4o
```

Run the benchmark:

```bash
uv run rubikbench run
```

`uv run rubikbench` does the same thing.

Without a config file, RubikBench uses the OpenAI endpoint and model by default. All other settings are optional. The supported `.env` variables are:

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | OpenAI key. RubikBench uses it automatically for an OpenAI endpoint. |
| `OPENROUTER_API_KEY` | OpenRouter key. RubikBench uses it automatically for an OpenRouter endpoint. |
| `RUBIKBENCH_API_KEY` | Generic API key override for any endpoint. |
| `RUBIKBENCH_BASE_URL` | Endpoint URL for any OpenAI-compatible `/v1` server. |
| `RUBIKBENCH_MODEL` | Model name. |
| `RUBIKBENCH_MAX_INPUT_TOKENS` | Context limit. RubikBench removes older turns when needed. |
| `RUBIKBENCH_MAX_OUTPUT_TOKENS` | Value that RubikBench sends as `max_tokens` in the request. |
| `RUBIKBENCH_TEMPERATURE` | Sampling temperature. A blank value uses the model default. |
| `RUBIKBENCH_TOP_P` | Nucleus sampling `top_p`. A blank value uses the model default. |
| `RUBIKBENCH_REPETITION_PENALTY` | vLLM-style repetition penalty. A blank value disables it. |
| `RUBIKBENCH_TOP_K` | vLLM-style top-k sampling. A blank value disables it. |
| `RUBIKBENCH_STREAM_IDLE_TIMEOUT` | Seconds without a response chunk before RubikBench aborts and retries the request. A blank value disables it. |
| `RUBIKBENCH_LOOP_DETECTION` | Set to `1` or `true` to enable loop detection (default). Set to `0` or `false` to disable it. |
| `RUBIKBENCH_TIMEOUT` | Request timeout in seconds. |
| `RUBIKBENCH_MAX_RETRIES` | Retries per request. |
| `RUBIKBENCH_SOLVES` | Number of solves. |
| `RUBIKBENCH_MAX_TURNS` | Turn budget for each solve. |
| `RUBIKBENCH_SCRAMBLE_LEN` | Scramble length. |
| `RUBIKBENCH_SEED` | Fixed scramble seed for reproducible runs. |

If a config file exists, environment values take precedence for the base URL, model, and API key. Other values come from the config file.

RubikBench resolves the API key in this order:

1. `RUBIKBENCH_API_KEY`
2. The provider variable that matches the endpoint URL or host, such as `OPENAI_API_KEY` or `OPENROUTER_API_KEY`
3. `OPENAI_API_KEY` for another OpenAI-compatible endpoint, such as a custom LiteLLM proxy

Local servers, including `localhost` and `*.local`, do not need a key. Run `uv run rubikbench validate` to check the resolved settings and the source of the key. Run `uv run rubikbench presets` to list providers.

List the presets:

```bash
uv run rubikbench presets
```

Check a configuration file:

```bash
uv run rubikbench validate --config cfg.json
```

Start a benchmark with a configuration file:

```bash
uv run rubikbench run --config cfg.json -o results.jsonl
```

The `run` command starts immediately. It does not need the TUI or a config file. Without `--config`, it reads `.env`. You can pass each setting as a flag. A flag overrides `.env` and the config file.

```bash
uv run rubikbench run \
  --base-url https://api.openai.com/v1 \
  --api-key sk-... \
  --model gpt-4o \
  --max-input-tokens 32768 \
  --max-output-tokens 4096 \
  -n 3 --max-turns 40 --scramble-len 22 --seed 42
```

Run `uv run rubikbench run --help` for the complete flag list. Model text and tool calls stream to the terminal during the run.

The `run` command prints a JSON summary to standard output. The summary includes solve rate, average score, average moves, token counts, retries, and truncated solves.

The command writes one line to the output file for each solve. The first line is a header with the configuration and aggregate statistics. Each solve line includes analytics and the complete transcript. Analytics include input tokens, output tokens, cached tokens, retries, finish reasons, and the truncation flag. Each solve line also includes a timeline with the cube state after each state change.

Merge run files into one dataset file:

```bash
uv run rubikbench aggregate run1.jsonl run2.jsonl -o dataset.json
```

The dataset file is one JSON document. It contains all solve records and totals. Totals include tokens, turns, tool calls, moves, retries, and truncated solves.

Replay a run file in the browser:

```bash
uv run rubikbench view results.jsonl
```

This command starts a local web server and opens the page in your browser. The page shows the cube in 3D. It replays the states in the timeline. You can play, pause, and jump to a step. You can also change the replay speed.

Optional flags:

- `--no-open` starts the server without opening the browser.
- `--port N` selects the port. The default port is 8321.

Press `Ctrl+C` to stop the server.

## 9. Run the tests

Run the test suite:

```bash
uv run pytest
```

The test suite covers the cube model, scramble generator, scoring, configuration, tool loop, token analytics, dataset aggregation, web replay, headless CLI, and TUI.

Check the style:

```bash
uv run ruff check
```

## 10. Project layout

```text
rubikbench/
  cube.py       # Cube model and move tables
  scramble.py   # Scramble generator
  rendering.py  # ASCII net and colored net
  scoring.py    # Benchmark score
  config.py     # Configuration values and presets
  llm.py        # OpenAI-compatible client
  benchmark.py  # Tool execution and turn loop
  prompts.py    # System prompt and tool schemas
  solver_ref.py # kociemba integration
  aggregate.py  # Dataset aggregation
  cli.py        # TUI entry point and headless commands
  webui/        # Web replay server and 3D page
  tui/          # Textual application
```
