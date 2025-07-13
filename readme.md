# ai-tools – README

Tiny toolkit for “chatting” with an entire codebase via OpenRouter or OpenAI.
```
~/ai-tools/
├── ai_cli.py   # directory-walker + chat client
└── Makefile    # one-line shortcuts
```


## 1  Install

```
pip install requests tiktoken tqdm
export OPENROUTER_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."      # only if you’ll call OpenAI directly

```

## 2 Quick Start
```
# scan the repo you’re in with GPT-4o (through OpenRouter)
make -f~/Documents/personal/ai-cli/Makefile g4o

# same, but hit OpenAI’s endpoint
make -f ~/Documents/personal/ai-cli/Makefile g4o PROVIDER=openai

# Claude Opus on another folder
make -f ~~/Documents/personal/ai-cli/Makefile opus DIR=/path/to/other/repo

```

## 3 Ad-hoc questions

```
make -f ~/Documents/personal/ai-cli/Makefile mistral PROMPT="Give me a concise summary of the mudah_scrape.py file only, ignore data folder"

```

Everything is optional—any var you omit falls back to the default in the Makefile.

## 4  Shortcut targets

```
| Target | Model slug                           |
|--------|--------------------------------------|
| `o3`      | `openai/o3`                       |
| `g41`     | `openai/gpt-4.1`                  |
| `g4o`     | `openai/chatgpt-4o-latest`        |
| `opus`    | `anthropic/claude-opus-4`         |
| `gemini`  | `google/gemini-2.5-pro`           |
| `mistral` | `mistralai/mistral-large-2411`    |
| `deepseek`| `deepseek/deepseek-r1-0528`       |
```
Add your own by editing `~/Documents/personal/ai-cli/Makefile`—copy a line and change the slug.



## 5  Per-repo convenience (optional)

Drop this one-liner into any project root:


include ~/ai-tools/Makefile

Now you can stay inside the repo and just run:
```
make g41
make ask p="List every file importing express."
```

## 6 Cost tracking
Every run ends with a summary:
```
Prompt tokens   : 18 732
Completion tok. :  4 109
Total tokens    : 22 841
Estimated cost  : $0.18
```


