# ADK compatibility evaluation

This one-case evaluation exercises the production agent's real
ADK → LiteLLM → OpenRouter → tool-call path. It requires an
`OPENROUTER_API_KEY` in the process environment; never add that key to the
repository or the command line.

Install the committed evaluation dependencies:

```bash
uv sync --locked --no-default-groups --group eval
```

The `eval` dependency group pins `google-adk[eval]==1.36.2`, including the
response-matching dependencies, in `uv.lock`.

Then expose `OPENROUTER_API_KEY` to the process environment and run:

```bash
ADK_DISABLE_LOAD_DOTENV=true \
GOOGLE_API_KEY= \
MEM0_EMBEDDER_DIMS= \
MEM0_EMBEDDER_MODEL=__disabled_for_adk_compatibility_eval__ \
OTEL_SDK_DISABLED=true \
PYTEST_ADDOPTS= \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTEST_PLUGINS= \
ROOT_AGENT_MODEL=google/gemini-2.5-flash \
uv run --locked --no-sync --no-default-groups --group eval \
  pytest --noconftest --confcutdir=tests/eval \
  -o addopts= -p no:cacheprovider \
  tests/eval/production_adk_eval.py \
  -q --tb=line --disable-warnings --show-capture=no
```

The Mem0 sentinel deliberately leaves the optional memory integration disabled
so this check isolates the framework and model tool-call boundary. The eval
must report one `example_tool` call, a `1.0` trajectory score, and a passing
response-match score. The gate parses ADK's ephemeral structured result and
requires both metrics to be explicitly `PASSED`; it also captures ADK's output
and fails unless the terminal summary contains exactly one passed case and zero
failed cases.
