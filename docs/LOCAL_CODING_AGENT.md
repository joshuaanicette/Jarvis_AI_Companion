# Local Embedded Coding Agent

This branch adds the first local-first Jarvis coding-agent foundation.

## Goals

- Route normal conversation to a small model.
- Route C/C++/embedded coding to Qwen Coder.
- Reuse Jarvis memory instead of retraining a model for every new fact.
- Keep file and shell access bounded.
- Require explicit approval before writes or destructive actions.
- Let models unload quickly to save RAM and power.

## Recommended Ollama models

```bash
ollama pull gemma3:1b
ollama pull qwen2.5:3b
ollama pull qwen2.5-coder:3b
ollama pull qwen2.5-coder:7b
```

Default routing:

- conversation -> `gemma3:1b`
- technical reasoning -> `qwen2.5:3b`
- coding/embedded -> `qwen2.5-coder:3b`
- complex coding/driver/RTOS/refactor -> `qwen2.5-coder:7b`

The 7B coder is optional on lower-memory devices. Change the router configuration
if only the 3B model should be used.

## Example

```python
from src.agent.coding_agent import CodingAgent

agent = CodingAgent(project_root=".")

result = agent.ask(
    "Explain why this UART driver may be dropping bytes.",
    category="embedded",
)

print(result.model)
print(result.response)
```

Inspect/search/build without write access:

```python
print(agent.inspect_file("src/main.c").output)
print(agent.search_code("HAL_UART").output)
print(agent.run_build(["make"]).output)
```

A file change is split into proposal and application:

```python
proposal = agent.propose_file_change(
    "src/main.c",
    "Add bounds checking to the receive buffer.",
)

# The caller/UI must explicitly confirm before this step.
result = agent.apply_file_change(
    "src/main.c",
    replacement_text,
    approved=True,
)
```

## Memory strategy

Keep knowledge outside model weights. Save verified engineering facts and successful
fixes in Jarvis memory, then retrieve only relevant items for each task.

Useful memory examples:

- board and MCU names
- toolchain/build commands
- register or pin mappings
- known project architecture
- recurring compiler/linker errors
- verified fixes
- preferred coding conventions

Large manuals and datasheets should later be added through retrieval/RAG rather than
copied into the system prompt.

## Power and RAM

Ollama requests use a short `keep_alive` period. This lets an idle model unload
instead of staying resident indefinitely. Use the 3B coder for most tasks and reserve
7B for harder work.
