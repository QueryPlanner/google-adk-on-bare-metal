# ADK compatibility evaluation

This one-case evaluation exercises the production agent's real
ADK → LiteLLM → OpenRouter → tool-call path. It requires an
`OPENROUTER_API_KEY` in the process environment; never add that key to the
repository or the command line.

Run it with:

```bash
ADK_DISABLE_LOAD_DOTENV=true \
GOOGLE_API_KEY= \
MEM0_EMBEDDER_DIMS= \
MEM0_EMBEDDER_MODEL=__disabled_for_adk_compatibility_eval__ \
OTEL_SDK_DISABLED=true \
ROOT_AGENT_MODEL=google/gemini-2.5-flash \
uv run --with 'google-adk[eval]==1.36.2' adk eval \
  src/agent tests/eval/adk_compatibility.evalset.json \
  --config_file_path=tests/eval/adk_compatibility.config.json \
  --print_detailed_results
```

The Mem0 sentinel deliberately leaves the optional memory integration disabled
so this check isolates the framework and model tool-call boundary. The eval
must report one `example_tool` call, a `1.0` trajectory score, and a passing
response-match score.
