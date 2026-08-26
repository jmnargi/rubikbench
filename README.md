# RubikBench

RubikBench tests how well a large language model (LLM) can solve a 3x3x3 Rubik's cube.

The model uses tools to observe the cube and to apply moves to it. The benchmark runs on its own. It connects to any OpenAI-compatible endpoint.

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
| Tool call | A request from the model to execute one tool. |
| Turn budget | The maximum number of conversation turns for one solve. |

## 1. Install

**Requirement:** Python 3.10 or newer.

Run this command:

```bash
uv sync --extra solver --extra dev
```

The solver extra installs kociemba. Kociemba gives the par value for scoring. It is optional. Without kociemba, RubikBench uses the value 20 for par.

## 2. Start the TUI

Run this command:

```bash
uv run rubikbench
```

## 3. Configure the benchmark

Do these steps on the configuration screen:

1. Select a preset. RubikBench has presets for OpenAI, OpenRouter, DeepSeek, Groq, Mistral, vLLM, Ollama, and LM Studio. Select "Apply preset" to fill the endpoint fields. For a custom server, enter the values yourself.
2. Enter the Base URL. It must be an OpenAI-compatible /v1 endpoint.
3. Enter the API key. Leave it empty for local servers, for example Ollama or vLLM.
4. Enter the model name.
5. Select the reasoning effort. The values are default, low, medium, and high. RubikBench sends this value in the request body.
6. Enter the max output tokens. Leave it empty to use the model default. RubikBench sends this value as "max_tokens" in the request body.
7. Enter the max input tokens. Leave it empty for no limit. RubikBench does not send this value to the server. It trims the older conversation turns to keep the request below this value.
8. Enter the extra body parameters in JSON. RubikBench sends them with every chat-completion request. Use this field for model-specific values, for example "max_completion_tokens".
9. Enter the benchmark values. The main values are:
  - The number of solves.
  - The turn budget.
  - The scramble length.
  - The seed. Leave it empty for a random seed.
10. Enter the scoring weights. The default weights are moves 0.5, conversation turns 0.3, and tool calls 0.2.
11. Select "Start benchmark".

RubikBench saves the configuration to the file `rubikbench_config.json` when the run starts.

## 4. Run the benchmark

The live run screen shows:

- The current cube, with colors.
- The move history.
- The statistics. The statistics are turns, tool calls, moves, elapsed time, and score.
- The log of the model activity.

You can abort the run. Use the "Abort" button.

## 5. How a solve works

For each solve, the benchmark does these steps:

1. Generate a scramble. The default length is 22 moves. The generator does not repeat the same face two times in a row.
2. Show the scramble to the model.
3. Ask the model for a reply. Each reply is one conversation turn.
4. Execute the tool calls in the reply.
5. Check the cube. If it is solved, the solve is complete.
6. Repeat steps 3 to 5 until one of these conditions is true:
   - The cube is solved.
   - The turn budget is empty.
   - An API error stops the run.
   - You abort the run.

The model has three tools:

- `get_cube_state` — returns the current state of the cube. It returns the 54 facelets, the colored net, and the move history.
- `apply_moves` — applies one or more moves in one call. The moves use Singmaster notation, for example "R U R' U'". Invalid tokens are not applied.
- `reset_cube` — returns the cube to the original scramble. It costs conversation turns. It does not add moves.

The model can apply many moves in one tool call. It can also make more than one tool call in one reply.

### Moves as text

Some models write moves as text instead of using the tool. RubikBench can parse these moves and apply them. The configuration value "allow text moves" controls this behavior. The default is on.

### Retries

If a request fails, RubikBench retries it. The number of retries is a configuration value. The default is 2.

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

Select "Export JSONL" to save the results. The output file is in the directory `rubikbench_results`.

## 8. Run the benchmark without the TUI

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

The run command prints a JSON summary to the standard output. The summary has the solve rate, the average score, the average moves, and the average turns.

The run command writes one line to the output file for each solve. The first line is the header. It contains the configuration and the aggregate statistics. Each solve line contains the full transcript.

## 9. Run the tests

Run this command:

```bash
uv run pytest
```

The test suite has 65 tests. It covers the cube model, the scramble generator, the scoring, the configuration, the tool loop, the headless CLI, and the TUI.

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
  cli.py        # The TUI entry point and the headless commands.
  tui/          # The Textual application.
```
