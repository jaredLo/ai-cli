# ---------- defaults ----------
MODEL    ?= openai/chatgpt-4o-latest
PROVIDER ?= openrouter
DIR      ?= .
PROMPT   ?= "Give me a concise summary of this file"
MAXTOK   ?= 16000

# NEW: expose the extra CLI knobs
INCLUDE_GLOB ?=
EXCLUDE_GLOB ?=
STOP_AFTER   ?=20
PREVIEW_LEN  ?=0

# ---------- generic target ----------
chat:
	python $(dir $(lastword $(MAKEFILE_LIST)))ai_cli.py \
	        --provider $(PROVIDER) \
	        --model $(MODEL) \
	        --dir $(DIR) \
	        --prompt "$(PROMPT)" \
	        --max-tokens $(MAXTOK) \
	        $(if $(INCLUDE_GLOB),--include-glob $(INCLUDE_GLOB)) \
	        $(if $(EXCLUDE_GLOB),--exclude-glob $(EXCLUDE_GLOB)) \
	        $(if $(STOP_AFTER),--stop-after $(STOP_AFTER)) \
	        $(if $(PREVIEW_LEN),--preview-len $(PREVIEW_LEN))

# ---------- shortcut aliases (no recursion) ----------
o3:      MODEL=openai/o3
o3:      chat

g41:     MODEL=openai/gpt-4.1
g41:     chat

g4o:     MODEL=openai/chatgpt-4o-latest
g4o:     chat

opus:    MODEL=anthropic/claude-opus-4
opus:    chat

gemini:  MODEL=google/gemini-2.5-pro
gemini:  chat

mistral: MODEL=mistralai/mistral-large-2411
mistral: chat

deepseek: MODEL=deepseek/deepseek-r1-0528
deepseek: chat

qwen: MODEL=qwen/qwen3-8b
qwen: chat

# helper for ad-hoc prompts
ask:     ; $(MAKE) chat PROMPT="$(p)"