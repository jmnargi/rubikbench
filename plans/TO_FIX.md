# RubikBench fixes and improvements

Work through these in numerical order. Change `[TODO]` to `[COMPLETE]` next to an item only after its implementation and focused verification pass. After every item, also run `uv run ruff check`; run the full `uv run pytest` suite before declaring the overall list complete.

1. [COMPLETE] **Correct inverted Singmaster move semantics — benchmark blocker**
   - The prompt defines `U R F D L B` as clockwise when looking at that face, but `rubikbench/cube.py` currently performs the opposite quarter turns.
   - Correct the permutation convention so unprimed moves are clockwise, primed moves are counterclockwise, and double moves remain unchanged.
   - Remove the compensating inversion in `rubikbench/solver_ref.py` and update expectations that currently encode the inverted behavior.
   - Add canonical, implementation-independent tests for all six faces plus reference-solver checks. Do not rely only on self-consistency tests such as move + inverse.
   - Verify with `uv run pytest tests/test_cube.py` and `uv run pytest`.

2. [COMPLETE] **Prevent API keys from being persisted or exported**
   - `BenchmarkConfig.to_dict()` currently includes `api_key`, and JSONL aggregation/replay paths preserve the serialized config.
   - Add a safe persistence/export representation that omits or redacts credentials while leaving in-memory request configuration functional.
   - Cover direct JSONL export, aggregation, and replay datasets.
   - Add a regression test asserting that a configured key is absent from every exported byte/string.
   - Any real key already included in a shared or committed result file should be considered exposed and rotated.

3. [COMPLETE] **Always preserve the latest cube state when trimming model context**
   - `trim_messages()` permanently keeps the system and initial user messages but can discard all newer tool results, leaving the model with the stale initial cube state.
   - Maintain a compact, synthetic latest-state checkpoint that survives trimming and explicitly supersedes earlier states.
   - Preserve the current facelet state and essential counters while dropping stale reasoning/tool history first.
   - Add tests proving that after multiple tool calls and aggressive trimming, the next request contains the latest cube state—not only the initial scramble—and still has valid assistant/tool-call structure.
   - Verify with the focused LLM/benchmark tests and the full suite.

4. [COMPLETE] **Implement honest context-window and output-token budgeting**
   - Separate the model's total context window from its output-token cap; rename ambiguous `max tokens` UI text to `output cap`.
   - Derive the available input budget as context window minus output reserve, tool/schema overhead, and a safety margin.
   - Include tool schemas and request overhead in estimates. Use an exact/provider tokenizer where supported and clearly label heuristic values as estimates otherwise.
   - Warn or fail clearly when the immutable prompt/current-state prefix cannot fit instead of silently exceeding the configured cap.
   - Add boundary tests for oversized prefixes, schema overhead, output reservation, and trimmed requests.

5. [COMPLETE] **Make move protocol behavior and scoring agree**
   - The system prompt says only `apply_moves` changes the cube, but `allow_text_moves` currently defaults to true and parsed assistant text can be applied.
   - Make strict tool-only mode the benchmark default; retain text-move behavior only as an explicitly named compatibility mode if it remains useful.
   - Record the protocol mode in every result/export so incompatible runs cannot be compared accidentally.
   - If text actions remain supported, count and score them as actions rather than awarding tool-discipline credit to a text-only solve.
   - Add tests for strict rejection/non-application, compatibility behavior, exported mode, and scoring.

6. [COMPLETE] **Make replay HTML safe for untrusted model transcripts**
   - Replay currently inserts `json.dumps(dataset)` directly into an inline script, allowing transcript text containing a script-closing sequence to escape the data block.
   - Embed data in an inert JSON element read through `textContent`, or perform equivalent robust escaping of HTML-significant delimiters.
   - Add a hostile-transcript regression test proving the content stays data and cannot create a second executable script block.
   - Preserve normal replay behavior and dataset shape.

7. [COMPLETE] **Correct reasoning, completion, and cache token analytics**
   - Persist reasoning-token data in `SolveResult`, JSONL exports, aggregation, CLI output, and TUI output instead of carrying it only in transient events.
   - Do not let streaming chunks incorrectly overwrite final/cumulative usage.
   - Display separate metrics for total generated/completion tokens, provider-reported reasoning tokens, estimated reasoning tokens, provider-reported cached input, and estimated cacheable shared prefix.
   - Persist whether each value was provider-reported or estimated; never present a cacheable-prefix estimate as a confirmed provider cache hit.
   - Add provider-shape and streaming tests covering missing details, nested details, cumulative usage, and fallback estimates.

8. [COMPLETE] **Add a reproducible difficulty ladder and capability-frontier report**
   - Add fixed benchmark sets around depths `1, 2, 3, 5, 8, 12`, plus the existing full/random difficulty.
   - Make every set deterministic/versioned so different models and runs receive identical states.
   - Report solve rate, efficiency, and failure boundary by difficulty instead of reducing evaluation to success/failure on a full scramble.
   - Keep the hidden scramble hidden from the model and validate every generated/fixed state as legal.
   - Add tests for deterministic selection, requested depth, result labeling, and aggregate frontier calculations.

9. [COMPLETE] **Add a versioned structured-cubie state presentation mode**
   - Keep the existing sticker/facelet mode as the strict spatial benchmark.
   - Add a separately labeled mode describing corner and edge pieces by current position and orientation, along with explicit face-orientation conventions and a guarantee that the state is legal.
   - Derive both sticker and cubie views from the same `Cube` state; never maintain two independent mutable cube representations.
   - Stamp the presentation mode/version into results and exports so modes cannot be compared as if equivalent.
   - Add solved-state, known-move, round-trip/invariant, prompt, and export tests.

10. [COMPLETE] **Add sane run profiles and useful TUI diagnostics**
    - Add named profiles such as smoke, diagnostic, full, and research while retaining explicit advanced overrides.
    - Use conservative diagnostic defaults (roughly 2k–8k output tokens and tens of turns) so models do not spend 64k tokens wandering without feedback.
    - Show current/peak context use, context limit, output reserve/cap utilization, finish reason, whether trimming occurred, protocol mode, tool calls versus text actions, and reported-versus-estimated token/cache metrics.
    - Add clear cube face-orientation labels in the TUI.
    - Warn about technically valid but uninformative configurations instead of silently changing user values.
    - Add config/profile precedence tests and focused TUI rendering/event tests.

11. [COMPLETE] **Remove stale reset-tool documentation and dead detection logic**
    - Remove the README claim that a `reset_cube` tool exists.
    - Remove the obsolete raw-tool detector branch for `reset_cube` from `rubikbench/benchmark.py`.
    - Search tests and user-facing help for other stale reset references and update only those tied to the removed tool.
    - Verify focused CLI/benchmark tests, then run the full suite.

## Required order

- Item 1 must be completed before interpreting new model performance or implementing the difficulty/presentation experiments in items 8–9.
- Items 3–4 should be completed before expensive long-context runs.
- Item 7 should be completed before using token/cache analytics for model comparisons.
- Items 2 and 6 are security fixes and should not be deferred behind product-facing enhancements.
