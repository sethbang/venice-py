# Venice AI CLI Reference

Command-line interface for Venice AI. Chat with AI models, generate images, create audio and video, manage embeddings, and more — all from your terminal.

The command is **`venice-py`**. It ships with this unofficial, community-maintained
Python SDK and is a different program from Venice's official CLI, which is written in
Node and invoked as `venice` — see [veniceai/venice-cli](https://github.com/veniceai/venice-cli)
for that one. The two can be installed side by side.

:::note Renamed in v2.1.0

This command was `venice` in v2.0.x, which collided with the official CLI's binary of the
same name. Update any scripts and aliases, and regenerate shell completions. Upgrading
does not delete the old entry point — if `venice --version` still prints a Venice AI CLI
banner, remove the leftover script with `rm "$(command -v venice)"` while the environment
is active.

:::

## Installation

### Prerequisites

- Python 3.13+
- A Venice AI API key ([venice.ai](https://venice.ai))

### Install with pip

Requires Python 3.13+. Quote the specifier — `[...]` is a glob in zsh.

```bash
pip install 'venice-py[cli]'
```

### Install with Poetry

```bash
poetry add 'venice-py[cli]'
```

### Development Install

```bash
git clone https://github.com/sethbang/venice-py.git
cd venice-py
poetry install --all-extras
poetry run venice-py --version
```

## Quick Start

```bash
# Set your API key
export VENICE_API_KEY="your-api-key"

# Or run the interactive setup wizard
venice-py configure

# Chat with an AI model
venice-py chat start "What is quantum computing?"

# Generate an image
venice-py image generate "A sunset over mountains"

# List available models
venice-py models
```

---

## Global Options

These options apply to the top-level `venice-py` command and affect all subcommands.

| Option | Description |
|--------|-------------|
| `--version` | Show version information and exit |
| `--plain` | Plain text output — no colors, panels, or emojis (see [Plain Mode](#plain-mode)) |
| `--config <path>` | Path to a custom config file (default: `~/.venice-py/config.yaml`) — see note below |
| `--help` | Show help text for any command |

```bash
venice-py --version
venice --plain chat start "Hello"
venice --config /path/to/config.yaml models
```

`--config <path>` is resolved for **all** subcommands: the chosen file's
`api.key` and `api.base_url` are applied to every client the CLI constructs.
The `VENICE_API_KEY` environment variable still takes precedence over
`api.key` in the file. `venice-py configure` reads from and writes back to the
same `--config` path, so `venice --config ./team.yaml configure` edits that
file rather than the default `~/.venice-py/config.yaml`.

---

## Configuration

### Interactive Setup

Run the configuration wizard to set your API key, default models, generation parameters, and output settings:

```bash
venice-py configure
```

The wizard walks through:

1. **API key** — enter or update your Venice AI API key
2. **Default models** — choose default models for chat, image, text-to-speech, speech-to-text, embedding, and video (text-to-video / image-to-video) from the live API
3. **Generation parameters** — set default temperature and max tokens
4. **Output settings** — configure the default image save directory
5. **Features** — enable or disable response streaming

### Config File

Configuration is stored at `~/.venice-py/config.yaml`:

```yaml
api:
  key: your-api-key-here          # optional if VENICE_API_KEY env var is set
  base_url: https://api.venice.ai/api/v1

defaults:
  chat_model: venice-uncensored
  image_model: hidream
  max_completion_tokens: 2048
  temperature: 0.7

output:
  format: markdown
  images_dir: ~/Pictures/venice

features:
  streaming: true
  cost_tracking: true   # reserved for a future feature; not read by any CLI command yet
```

Saved chat transcripts live in `~/.venice-py/conversations/` and image presets in
`~/.venice-py/presets/`.

:::note Upgrading from `~/.venice/`

Earlier releases kept this data in `~/.venice/`, which collides with the official
`venice` CLI. The first `venice-py` command that reads the directory copies
`config.yaml`, `conversations/` and `presets/` across and prints a one-line notice.
The old directory is left exactly as it was, so both tools keep working; delete it
yourself once you are satisfied nothing else needs it.

:::

### Environment Variables

| Variable | Description |
|----------|-------------|
| `VENICE_API_KEY` | Your Venice AI API key (takes precedence over config file) |

---

## Shell Completion

Generate shell completion scripts for tab-completion of commands and options.

```bash
# Bash
venice-py completion bash >> ~/.bashrc

# Zsh
venice-py completion zsh >> ~/.zshrc

# Fish
venice-py completion fish > ~/.config/fish/completions/venice.fish
```

---

## Chat

Chat with AI models via streaming or one-shot messages. Supports interactive sessions, conversation history, piped input, Venice-specific parameters, and JSON output.

### `venice-py chat start`

Start an interactive chat session or send a single message.

```bash
venice-py chat start [MESSAGE] [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `MESSAGE` | Optional message to send. If omitted, starts an interactive session. |

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | runtime * | AI model to use |
| `--select-model` | `-S` | — | Interactively select a model from the API |
| `--system` | `-s` | — | System prompt to set AI personality |
| `--temperature` | `-t` | `0.7` ² | Temperature for response generation |
| `--max-completion-tokens` | | `2048` ¹ | Maximum tokens in response |
| `--stream / --no-stream` | | `--stream` | Enable or disable streaming |
| `--show-thinking / --hide-thinking` | | `--hide-thinking` | Show reasoning process for reasoning models |
| `--animation` | `-a` | `smooth` | Animation style: `none`, `smooth`, `word`, `char`, `line`, `typewriter` |
| `--animation-speed` | | `0.03` | Animation speed in seconds (lower = faster) |
| `--show-stats / --hide-stats` | | `--hide-stats` | Show streaming statistics after response |
| `--top-p` | | — | Top-p (nucleus) sampling parameter (0.0–1.0) |
| `--web-search` | | — | Enable web search: `auto`, `on`, `off` |
| `--character` | | — | Character slug for character-driven chat |
| `--reasoning-effort` | | — | Reasoning effort tier: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` (per-model support) |
| `--strip-thinking / --no-strip-thinking` | | — | Strip thinking/reasoning from response |
| `--disable-thinking` | | `false` | Disable thinking/reasoning entirely |
| `--json-output` | | `false` | Output the full API response as JSON |
| `--save` | | `false` | Save conversation after session ends |
| `--continue-from` | | — | Continue a previous conversation by ID (use `last` for most recent) |

> \* No static default — resolved at runtime: `--model` wins, else a matching `defaults.chat_model` in your config if set, else the live API-recommended model.
>
> ¹ The `--max-completion-tokens` flag has no built-in default (the click option defaults to unset); `2048` reflects the example configuration. The value actually used falls back to `defaults.max_completion_tokens` in your `~/.venice-py/config.yaml` when the flag is omitted.
>
> ² The `--temperature` flag has no built-in default (the click option defaults to unset); `0.7` reflects the example configuration. The value actually used falls back to `defaults.temperature` in your `~/.venice-py/config.yaml` when the flag is omitted.

#### Examples

```bash
# Interactive session
venice-py chat start

# Single message
venice-py chat start "Explain the theory of relativity"

# Choose a specific model
venice-py chat start --model qwen3-235b "Summarize this article"

# System prompt with custom temperature
venice-py chat start --system "You are a Python tutor" --temperature 0.3

# Show reasoning process
venice-py chat start --show-thinking "Solve: what is 23 * 47?"

# Typewriter animation with stats
venice-py chat start --animation typewriter --show-stats "Write a haiku"

# Web search enabled
venice-py chat start --web-search on "What happened in tech news today?"

# Chat as a character
venice-py chat start --character alan-watts "What is the meaning of life?"

# Reasoning model with effort control
venice-py chat start --model deepseek-r1-0528 --reasoning-effort high "Prove P != NP"

# Non-streaming JSON output (for scripting)
venice-py chat start --no-stream --json-output "What is 2+2?"

# Save and resume conversations
venice-py chat start --save "Tell me about quantum computing"
venice-py chat start --continue-from last "What are its applications?"

# Pipe input from stdin
echo "Summarize this text" | venice-py chat start
cat article.txt | venice-py chat start --model llama-3.3-70b "Summarize:"
```

### `venice-py chat history`

View and manage saved conversations.

```bash
venice-py chat history [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | `false` | Output conversation list as JSON |
| `--delete <id>` | — | Delete a conversation by ID |

```bash
# List saved conversations
venice-py chat history

# Export as JSON
venice-py chat history --json

# Delete a conversation
venice-py chat history --delete abc12345
```

---

## Image

Full-featured image generation, upscaling, editing, background removal, batch generation, style management, and presets.

### `venice-py image generate`

Generate an image from a text prompt.

```bash
venice-py image generate [PROMPT] [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | runtime * | Image model to use |
| `--size` | `-s` | `1024x1024` | Image size in WxH format (64–4096px per dimension) |
| `--num-images` | `-n` | `1` | Number of images to generate (1–4) |
| `--output` | `-o` | auto-generated | Output filename (without extension) |
| `--save-dir` | | `~/Pictures/venice` | Directory to save images |
| `--steps` | | model default | Inference steps (1–50, higher = better quality) |
| `--cfg-scale` | | model default | CFG scale (0–20, higher = stricter prompt adherence) |
| `--seed` | | random | Random seed for reproducibility |
| `--style-preset` | `-sp` | — | Style preset to apply (see `venice-py image list-styles`) |
| `--lora-strength` | | — | LoRA strength 0–100 |
| `--format` | | `png` | Output format: `jpeg`, `png`, `webp` |
| `--aspect-ratio` | | — | Aspect ratio for the output (e.g. `1:1`, `16:9`) |
| `--resolution` | | — | Resolution tier: `1K`, `2K`, `4K` |
| `--quality` | | — | Output quality for quality-aware models: `low`, `medium`, `high` |
| `--enable-web-search / --no-enable-web-search` | | — | Allow the image model to incorporate web search results |
| `--safe-mode / --no-safe-mode` | | — | Blur adult content |
| `--hide-watermark` | | `false` | Hide the Venice watermark |
| `--embed-exif` | | `false` | Embed generation metadata in EXIF |
| `--return-binary` | | `false` | Return raw binary data (faster) |
| `--show-timing / --no-show-timing` | | `--show-timing` | Show generation timing |
| `--preset` | `-p` | — | Load a saved or built-in preset |
| `--interactive` | `-i` | `false` | Launch the interactive wizard |
| `--open` | | `false` | Open image after saving |

> \* No static default — resolved at runtime: `--model` wins, else a matching `defaults.image_model` in your config if set, else the live API-recommended model.

#### Examples

```bash
# Basic generation
venice-py image generate "A serene mountain landscape at sunset"

# Specific model and size
venice-py image generate "Cyberpunk city" --model flux-dev --size 1280x720

# Fine-tuned generation
venice-py image generate "Portrait of a scientist" \
  --steps 40 --cfg-scale 7.5 --style-preset photographic \
  --format png --embed-exif

# Reproducible generation with seed
venice-py image generate "Cyberpunk cityscape" --seed 12345

# Aspect ratio, high-resolution tier, and quality
venice-py image generate "Editorial product shot" \
  --aspect-ratio 16:9 --resolution 4K --quality high

# Let the model pull in web search results
venice-py image generate "Today's top headline as an infographic" --enable-web-search

# Multiple images
venice-py image generate "Abstract art" --num-images 3 --save-dir ./art

# Using a preset
venice-py image generate "Fantasy landscape" --preset photorealistic

# Interactive wizard
venice-py image generate --interactive

# Open in viewer after generation
venice-py image generate "Cute cat" --open
```

### `venice-py image upscale`

Upscale an image using AI.

```bash
venice-py image upscale <INPUT_FILE> [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--scale` | | `2` | Scale factor — must be `2` or `4` |
| `--creativity` | | — | Detail and texture the upscaler adds; the server clamps this to 0–0.02 (default 0.01) |
| `--output` | `-o` | auto-generated | Output file path |
| `--save-dir` | | `.` | Directory to save the result |
| `--open` | | `false` | Open image after saving |

```bash
venice-py image upscale photo.jpg --scale 2
venice-py image upscale photo.png --scale 4 --creativity 0.02 --save-dir ./upscaled
venice-py image upscale logo.png --output logo_hd.png --open
```

### `venice-py image edit`

Edit an image using a text prompt.

```bash
venice-py image edit <INPUT_FILE> --prompt <INSTRUCTION> [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--prompt` | `-p` | **(required)** | Edit instruction describing what to change |
| `--model` | `-m` | — | Edit model to use |
| `--output` | `-o` | auto-generated | Output file path |
| `--save-dir` | | `.` | Directory to save the result |
| `--format` | | input format | Filename-extension hint: `png`, `jpeg`, `webp` |
| `--aspect-ratio` | | — | Aspect ratio for the output (e.g. `1:1`, `16:9`) |
| `--resolution` | | — | Resolution tier: `1K`, `2K`, `4K` |
| `--output-format` | | resolution-inferred | Output format sent to the API: `jpeg`, `png`, `webp` |
| `--safe-mode / --no-safe-mode` | | — | Blur adult content (server default on) |
| `--open` | | `false` | Open image after saving |

> `--output-format` is sent to the API and sets the real encoding; `--format` only hints the saved filename extension. When both are given, `--output-format` wins for the extension too.

```bash
venice-py image edit photo.jpg --prompt "Add a rainbow to the sky"
venice-py image edit portrait.png --prompt "Change hair color to red"
venice-py image edit scene.jpg -p "Make it nighttime" --format png --open
venice-py image edit photo.jpg -p "Editorial retouch" \
  --aspect-ratio 16:9 --resolution 4K --output-format jpeg
venice-py image edit photo.jpg -p "Add a rainbow" --no-safe-mode
```

### `venice-py image multi-edit`

Edit an image using up to three layered inputs. The first image is the base; additional images are layered on top, guided by a single prompt.

```bash
venice-py image multi-edit --prompt <INSTRUCTION> --image <BASE_FILE> [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--prompt` | `-p` | **(required)** | Edit instruction describing the changes |
| `--image` | `-i` | **(required)** | Base image (first layer) |
| `--image-2` | | — | Second layer image |
| `--image-3` | | — | Third layer image |
| `--model` | `-m` | — | Edit model to use |
| `--output` | `-o` | auto-generated | Output file path |
| `--save-dir` | | `.` | Directory to save the result |
| `--aspect-ratio` | | — | Aspect ratio for the output (e.g. `1:1`, `16:9`) |
| `--resolution` | | — | Resolution tier: `1K`, `2K`, `4K` |
| `--output-format` | | input format | Output format: `jpeg`, `png`, `webp` |
| `--quality` | | — | Output quality for quality-aware models: `low`, `medium`, `high` |
| `--open` | | `false` | Open image after saving |

```bash
venice-py image multi-edit --prompt "Combine these into one scene" --image base.png
venice-py image multi-edit -p "Blend the overlay onto the base" -i base.png --image-2 overlay.png
venice-py image multi-edit -p "Composite three layers" -i base.png --image-2 mid.png --image-3 top.png \
  --output-format png --quality high
```

### `venice-py image remove-bg`

Remove the background from an image (returns transparent PNG by default).

```bash
venice-py image remove-bg <INPUT_FILE> [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | auto-generated | Output file path |
| `--save-dir` | | `.` | Directory to save the result |
| `--format` | | `png` | Output format: `png` (recommended), `jpeg`, `webp` |
| `--open` | | `false` | Open image after saving |

```bash
venice-py image remove-bg photo.jpg
venice-py image remove-bg product.png --save-dir ./cutouts --open
venice-py image remove-bg headshot.jpg --output headshot_nobg.png
```

### `venice-py image batch`

Generate multiple images from a file of prompts (one prompt per line).

```bash
venice-py image batch --prompts-file <FILE> [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--prompts-file` | `-f` | **(required)** | File containing prompts, one per line |
| `--model` | `-m` | runtime * | Image model to use |
| `--size` | `-s` | `1024x1024` | Image size in WxH format |
| `--save-dir` | | `~/Pictures/venice` | Directory to save images |
| `--steps` | | model default | Inference steps |
| `--cfg-scale` | | model default | CFG scale |
| `--seed` | | random | Random seed |
| `--style-preset` | `-sp` | — | Style preset for all images |
| `--format` | | `png` | Output format: `jpeg`, `png`, `webp` |
| `--safe-mode / --no-safe-mode` | | — | Blur adult content |
| `--hide-watermark` | | `false` | Hide Venice watermark |
| `--embed-exif` | | `false` | Embed generation metadata in EXIF |

> \* No static default — resolved at runtime: `--model` wins, else a matching `defaults.image_model` in your config if set, else the live API-recommended model.

```bash
# Create a prompts file
cat > prompts.txt << 'EOF'
A serene mountain lake at dawn
A bustling Tokyo street at night
A peaceful English garden in spring
EOF

# Generate all with consistent style
venice-py image batch --prompts-file prompts.txt --style-preset cinematic --steps 30
```

### `venice-py image list-styles`

List all available style presets from the API.

```bash
venice-py image list-styles
```

Common styles include: `photographic`, `cinematic`, `anime`, `fantasy-art`, `digital-art`, `3d-model`, `pixel-art`, and more.

### `venice-py image presets`

Launch the interactive preset manager. Create, view, and delete image generation presets.

```bash
venice-py image presets
```

The interactive menu offers:

- **List all presets** — view your custom presets
- **View built-in presets** — see the 5 built-in configurations
- **Save current config as preset** — create a new preset interactively
- **Delete a preset** — remove a custom preset

#### Built-in Presets

| Preset | Steps | CFG | Format | Description |
|--------|-------|-----|--------|-------------|
| `photorealistic` | 30 | 7.5 | png | High-quality photorealistic images |
| `artistic` | 25 | 9.0 | webp | Artistic and creative styles |
| `quick` | 15 | 7.0 | webp | Fast generation with good quality |
| `high-quality` | 50 | 8.0 | png | Maximum quality (slower) |
| `creative` | 20 | 5.0 | webp | More creative freedom, less prompt adherence |

Custom presets are stored in `~/.venice-py/presets/` as JSON files.

```bash
# Apply a preset during generation
venice-py image generate "Portrait photo" --preset photorealistic

# Override specific preset values
venice-py image generate "Abstract art" --preset artistic --steps 40
```

---

## Audio

Text-to-speech and speech-to-text capabilities.

### `venice-py audio speak`

Convert text to speech.

```bash
venice-py audio speak [TEXT] [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | runtime * | TTS model to use |
| `--voice` | `-v` | `af_heart` | Voice to use |
| `--format` | | `mp3` | Output format: `mp3`, `wav`, `opus`, `flac`, `aac`, `pcm` |
| `--speed` | | `1.0` | Speech speed (0.25–4.0) |
| `--output` | `-o` | `speech_<timestamp>.<format>` | Output file path |
| `--save-dir` | | `.` | Directory to save the audio file |

> \* No static default — resolved at runtime: `--model` wins, else a matching `defaults.tts_model` in your config if set, else the live API-recommended model.

```bash
# Basic text-to-speech
venice-py audio speak "Hello, welcome to Venice AI!"

# Custom voice and format
venice-py audio speak "Good morning" --voice af_heart --format wav

# Adjust speed and save location
venice-py audio speak "This is a fast narration" --speed 1.5 --output narration.mp3

# Pipe text from stdin
echo "Read this aloud" | venice-py audio speak
cat script.txt | venice-py audio speak --output script_audio.mp3
```

### `venice-py audio transcribe`

Transcribe audio to text.

```bash
venice-py audio transcribe <FILE> [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | runtime * | STT model to use |
| `--language` | `-l` | auto-detect | Language code (e.g., `en`, `es`, `fr`) |
| `--timestamps` | | `false` | Include word-level timestamps |
| `--format` | | `text` | Output format: `text`, `json` |
| `--output` | `-o` | stdout | Save transcription to file |

> \* No static default — resolved at runtime: `--model` wins, else a matching `defaults.stt_model` in your config if set, else the live API-recommended model.

```bash
# Basic transcription
venice-py audio transcribe recording.mp3

# With language hint and timestamps
venice-py audio transcribe meeting.wav --language en --timestamps

# Save to file
venice-py audio transcribe interview.mp3 --output transcript.txt

# JSON output
venice-py audio transcribe audio.mp3 --format json
```

### `venice-py audio voices`

List available TTS voices. Wraps `Audio.get_voices()`. Supports filtering
by gender, region, and model.

```bash
venice-py audio voices
venice-py audio voices --gender female
venice-py audio voices --region af --json
venice-py audio voices --model tts-kokoro
```

---

## Video

Generate videos from text prompts or animate existing images. Video generation is asynchronous — jobs are queued and polled until complete.

### `venice-py video generate`

Generate a video from a text prompt.

```bash
venice-py video generate <PROMPT> [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | runtime * | Video model to use |
| `--duration` | | `5s` | Duration (e.g., `5s`, `10s`) |
| `--resolution` | | — | Output resolution (e.g., `720p`, `1080p`) |
| `--aspect-ratio` | | `16:9` | Aspect ratio (e.g., `16:9`, `9:16`, `1:1`) |
| `--negative-prompt` | | — | Content to avoid in the video |
| `--audio / --no-audio` | | — | Generate audio if the model supports it (omitted from the request unless set) |
| `--reference-image-urls` | | — | Reference image URL for character/style consistency (repeatable, up to 9) |
| `--reference-video-urls` | | — | Reference video URL for R2V models (repeatable, up to 3) |
| `--reference-audio-urls` | | — | Reference audio donor URL for R2V models (repeatable, up to 3) |
| `--end-image-url` | | — | End-frame image URL for models that support transitions |
| `--output` | `-o` | `video_<timestamp>.mp4` | Output file path |
| `--save-dir` | | `.` | Directory to save the video |
| `--no-poll` | | `false` | Queue the job and return the job ID without waiting |
| `--open` | | `false` | Open the video after saving |

> \* No static default — resolved at runtime: `--model` wins, else a matching `defaults.video_t2v_model` in your config if set, else the live API-recommended model.
>
> The CLI exposes the flat reference/transition fields above. The structured `elements[]` (per-segment prompts) and `scene_image_urls` parameters remain SDK-only — pass them through `client.video.run()`/`submit()`.

```bash
# Basic video generation
venice-py video generate "A sunset over the ocean with gentle waves"

# Custom duration and aspect ratio
venice-py video generate "A cat playing piano" --duration 10s --aspect-ratio 1:1

# Cinematic high-res
venice-py video generate "Cinematic drone shot over mountains" --resolution 1080p

# Queue without waiting
venice-py video generate "Quick preview scene" --no-poll

# Reference images for character consistency, plus audio
venice-py video generate "She walks toward the camera" \
  --audio --reference-image-urls https://example.com/ref1.png

# Transition to an end frame
venice-py video generate "Morning to night timelapse" \
  --end-image-url https://example.com/night.png
```

### `venice-py video from-image`

Animate an image into a video. The prompt describes the desired motion, not the image content.

```bash
venice-py video from-image <INPUT_FILE> [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--prompt` | `-p` | `""` | Motion prompt to guide animation |
| `--model` | `-m` | runtime * | Image-to-video model |
| `--duration` | | `5s` | Duration (e.g., `5s`, `10s`) |
| `--resolution` | | — | Output resolution |
| `--audio / --no-audio` | | — | Generate audio if the model supports it (omitted from the request unless set) |
| `--reference-image-urls` | | — | Reference image URL for character/style consistency (repeatable, up to 9) |
| `--reference-video-urls` | | — | Reference video URL for R2V models (repeatable, up to 3) |
| `--reference-audio-urls` | | — | Reference audio donor URL for R2V models (repeatable, up to 3) |
| `--end-image-url` | | — | End-frame image URL for models that support transitions |
| `--output` | `-o` | `video_<timestamp>.mp4` | Output file path |
| `--save-dir` | | `.` | Directory to save the video |
| `--no-poll` | | `false` | Queue without waiting |
| `--open` | | `false` | Open the video after saving |

> \* No static default — resolved at runtime: `--model` wins, else a matching `defaults.video_i2v_model` in your config if set, else the live API-recommended model.
>
> As with `generate`, structured `elements[]` and `scene_image_urls` parameters remain SDK-only.

```bash
# Animate an image with default motion
venice-py video from-image landscape.jpg

# Describe the motion
venice-py video from-image photo.png --prompt "Gentle breeze through the trees"

# Longer duration
venice-py video from-image portrait.jpg --duration 10s --open
```

### `venice-py video status`

Check the status of a video generation job (useful with `--no-poll`).

```bash
venice-py video status <JOB_ID> [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | — | Model used when the job was queued |

> No default is applied here; pass the same model the job was queued with.

```bash
venice-py video status abc123
venice-py video status abc123 --model wan-2.6-image-to-video
```

---

## Characters

Browse and inspect pre-configured AI personalities and specialized assistants.

### `venice-py characters list`

List available AI characters.

```bash
venice-py characters list [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | `false` | Output as JSON |
| `--search` | — | Free-text search across name, description, tags, and hashtags. Server-side — the API searches the full catalog rather than filtering a locally fetched first page. |
| `--sort-by` | — | Sort mode: `featured`, `highestRating`, `highlyRated`, `highlyRatedAndRecent`, `imports`, `mostRecent`, `ratingCount` |
| `--sort-order` | — | Sort order applied to `--sort-by`: `asc`, `desc` |
| `--tags` | — | Filter by tag name (repeatable) |
| `--categories` | — | Filter by category name (repeatable) |
| `--limit` | — | Number of characters to return |
| `--offset` | — | Number of characters to skip (for paging) |
| `--adult / --no-adult` | — | Filter by the adult-content flag |
| `--pro / --no-pro` | — | Filter to characters using pro models |
| `--web-enabled / --no-web-enabled` | — | Filter to web-enabled characters |
| `--model-id` | — | Filter by model ID |

```bash
venice-py characters list
venice-py characters list --search coding
venice-py characters list --sort-by highlyRated --limit 20
venice-py characters list --tags philosophy --web-enabled
venice-py characters list --categories education
venice-py characters list --json
```

### `venice-py characters info`

Show detailed information about a specific character.

```bash
venice-py characters info <SLUG> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | `false` | Output as JSON |

The human-readable view shows the slug, model, description, tags, adult/web flags, import count, and timestamps. `--json` emits the full record — `id`, `author`, `featured`, `isOwner`, `shareUrl`, `photoUrl`, `createdAt`/`updatedAt`, and a nested `stats` object (`imports`, `averageRating`, `ratingCount`, `ratingSum`, `userRating`).

```bash
venice-py characters info alan-watts
venice-py characters info alan-watts --json
```

Use a character in chat with `--character`:

```bash
venice-py chat start --character alan-watts "What is the meaning of life?"
```

### `venice-py characters reviews`

Show public reviews for a character.

```bash
venice-py characters reviews <SLUG> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | `false` | Output as JSON |
| `--page` | — | 1-indexed page number |
| `--page-size` | — | Reviews per page |

The summary line reports the average rating and total review count; each review shows its rating, creation date, and message.

```bash
venice-py characters reviews alan-watts
venice-py characters reviews alan-watts --page 2 --page-size 50
venice-py characters reviews alan-watts --json
```

---

## Embeddings

Generate text embeddings for semantic analysis, similarity search, and clustering.

```bash
venice-py embeddings [TEXT] [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | runtime * | Embedding model to use |
| `--encoding-format` | | `float` | Encoding format: `float`, `base64` |
| `--dimensions` | | model default | Number of output dimensions |
| `--json` | | `false` | Output full JSON response with embedding vector |
| `--output` | `-o` | stdout | Save embeddings to file |

> \* No static default — resolved at runtime: `--model` wins, else a matching `defaults.embedding_model` in your config if set, else the live API-recommended model.

```bash
# Generate embeddings
venice-py embeddings "The quick brown fox"

# Full JSON output with vector
venice-py embeddings "Hello world" --json

# Save to file
venice-py embeddings "Some text" --output embeddings.json

# Pipe text from stdin
echo "Embed this text" | venice-py embeddings

# Custom dimensions
venice-py embeddings "Search query" --dimensions 256
```

---

## Models

List, filter, compare, and inspect available AI models. The `models` command supports extensive filtering by type, capabilities, price, and status.

```bash
venice-py models [OPTIONS]
```

### Display Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--verbose` | `-v` | `false` | Show detailed information for all models |
| `--json` | | `false` | Output as JSON for scripting |
| `--currency` | | `both` | Currency to display: `usd`, `diem`, `both` |
| `--no-legend` | | `false` | Hide the capability icon legend |

### Type Filtering

| Option | Short | Description |
|--------|-------|-------------|
| `--type` | `-t` | Filter by type — can be repeated. Choices: `text`, `image`, `embedding`, `tts`, `asr`, `music`, `upscale`, `inpaint`, `video`. An unknown value errors instead of returning nothing. |

### Capability Filtering

| Option | Description |
|--------|-------------|
| `--function-calling` | Models with function calling support |
| `--vision` | Models with vision capabilities |
| `--reasoning` | Models with reasoning support |
| `--web-search` | Models with web search |
| `--code` | Models optimized for code |
| `--response-schema` | Models supporting response schemas |

### Trait Filtering

| Option | Description |
|--------|-------------|
| `--trait <value>` | Filter by trait (repeatable) |

### Price Filtering

| Option | Description |
|--------|-------------|
| `--max-input <price>` | Maximum input price per 1M tokens (USD) |
| `--max-output <price>` | Maximum output price per 1M tokens (USD) |
| `--max-gen <price>` | Maximum generation price (for images) |
| `--budget <price>` | Maximum average price (input+output)/2 |

### Status Filtering

| Option | Description |
|--------|-------------|
| `--beta` | Show only beta models |
| `--no-beta` | Exclude beta models |
| `--online` | Show only online models |

### Search & Detail

| Option | Description |
|--------|-------------|
| `--search <query>` | Search in model names and descriptions |
| `--id <model-id>` | Show details for a specific model |
| `--compare <id1,id2,...>` | Side-by-side comparison of models (comma-separated) |

### Sorting

| Option | Default | Values |
|--------|---------|--------|
| `--sort` | `name` | `name`, `id`, `price-asc`, `price-desc`, `context`, `created` |

### Examples

```bash
# List all models
venice-py models

# Text models only, sorted by price
venice-py models --type text --sort price-asc

# Image models
venice-py models --type image

# Models with vision and function calling
venice-py models --vision --function-calling

# Budget-friendly models under $1/M tokens
venice-py models --budget 1.0

# Exclude beta models
venice-py models --no-beta

# Search for a model
venice-py models --search "llama"

# Detailed view of a specific model
venice-py models --id qwen3-235b

# Compare two models side-by-side
venice-py models --compare "qwen3-235b,llama-3.3-70b"

# JSON output for scripting
venice-py models --json

# Verbose listing with full per-model detail
venice-py models --verbose
```

### `venice-py models resolve`

Auto-pick a model by type and capabilities. Wraps
`Models.resolve()` and the ten typed `resolve_*()` helpers
(`resolve_chat`, `resolve_image`, `resolve_embedding`, etc.). Useful in
scripts when you want "the cheapest model that supports vision + function
calling" without hardcoding a specific model ID.

```bash
venice-py models resolve --type chat --function-calling
venice-py models resolve --type image
venice-py models resolve --type chat --vision --min-context-tokens 32000
```

### `venice-py models get`

Show the full model record for a single ID. Wraps `Models.get()`. Pass
`--json` for machine-readable output.

```bash
venice-py models get llama-3.3-70b
venice-py models get qwen3-235b --json
```

### `venice-py models capabilities`

Show the structured `Capabilities` for a single model
(`Models.get_capabilities()`). Includes function-calling, vision,
reasoning, web-search, code, response-schema, and TEE/e2ee flags.

```bash
venice-py models capabilities llama-3.3-70b
venice-py models capabilities qwen3-235b --json
```

---

## Account

View account billing information and usage statistics.

### `venice-py account balance`

Show current account balance and credits.

```bash
venice-py account balance [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | `false` | Output as JSON |

```bash
venice-py account balance
venice-py account balance --json
```

### `venice-py account usage`

Show usage statistics for your account. Displays billing entries, costs, and a breakdown by product SKU.

```bash
venice-py account usage [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | `false` | Output as JSON |
| `--start-date` | — | Start date (`YYYY-MM-DD` or ISO 8601) |
| `--end-date` | — | End date (`YYYY-MM-DD` or ISO 8601) |
| `--currency` | — | Filter entries by currency: `USD`, `DIEM`, `VCU`, `BUNDLED_CREDITS` |

```bash
venice-py account usage
venice-py account usage --start-date 2025-01-01 --end-date 2025-02-01
venice-py account usage --currency USD
venice-py account usage --json
```

### `venice-py account keys get`

Retrieve a single API key by ID. Wraps `client.api_keys.retrieve()`. The
top-level `venice-py api-keys` group covers list/create/delete; the
`account keys` subgroup adds the single-key retrieve / update surface.

```bash
venice-py account keys get key_abc123
venice-py account keys get key_abc123 --json
```

### `venice-py account keys update`

Update an existing API key. Wraps `client.api_keys.update()`. At least one
of `--description`, `--expiry`, `--limit-usd`, `--limit-diem`, `--limit-vcu`,
or `--limit-period` must be provided.

| Option | Description |
|--------|-------------|
| `--description` | New description for the key |
| `--expiry` | New expiration (ISO 8601, e.g. `2026-12-31`) |
| `--limit-usd` | USD consumption limit |
| `--limit-diem` | DIEM consumption limit |
| `--limit-vcu` | VCU consumption limit (legacy alias for diem) |
| `--limit-period` | Period over which the consumption limit resets: `EPOCH`, `MONTH`, `LIFETIME` |
| `--json` | Output JSON |

```bash
venice-py account keys update key_abc123 --description "Production"
venice-py account keys update key_abc123 --expiry 2026-12-31
venice-py account keys update key_abc123 --limit-usd 50 --limit-diem 5
venice-py account keys update key_abc123 --limit-period MONTH
```

### `venice-py account keys rate-limits`

Show the current rate limits (per-model RPM / RPD / TPM) for your API key.
Wraps `client.api_keys.get_rate_limits()`. The header reports your API tier,
whether access is permitted, and when the next epoch begins.

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | `false` | Output as JSON |

```bash
venice-py account keys rate-limits
venice-py account keys rate-limits --json
```

### `venice-py account keys rate-limit-logs`

Show the most recent rate-limit violations for your account (up to 50).
Wraps `client.api_keys.get_rate_limit_logs()`. Each entry lists a timestamp,
model, limit type, and tier.

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | `false` | Output as JSON |

```bash
venice-py account keys rate-limit-logs
venice-py account keys rate-limit-logs --json
```

---

## API Keys

Manage your Venice AI API keys. List, create, and delete keys for your account.

### `venice-py api-keys list`

List all API keys.

```bash
venice-py api-keys list [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | `false` | Output as JSON |

```bash
venice-py api-keys list
venice-py api-keys list --json
```

### `venice-py api-keys create`

Create a new API key. The secret key is only shown once — save it immediately.

```bash
venice-py api-keys create --name <NAME> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--name` (alias: `--description`) | **(required)** | Name / description for the new key |
| `--type` | `INFERENCE` | API key type: `INFERENCE`, `ADMIN` |
| `--limit-usd` | — | USD consumption limit |
| `--limit-diem` | — | DIEM consumption limit |
| `--limit-vcu` | — | VCU consumption limit (legacy alias for `--limit-diem`; folds into `diem` unless `--limit-diem` is also given) |
| `--limit-period` | — | Period over which the consumption limit resets: `EPOCH`, `MONTH`, `LIFETIME` |
| `--expiry` | — | Expiration date (ISO 8601, e.g. `2026-12-31` or `2026-12-31T23:59:00Z`) |
| `--json` | `false` | Output as JSON |

```bash
venice-py api-keys create --name "Production Key"
venice-py api-keys create --name "Admin" --type ADMIN
venice-py api-keys create --name "Capped" --limit-usd 50 --limit-period MONTH
venice-py api-keys create --name "Expiring" --expiry 2026-12-31
venice-py api-keys create --name "CI/CD Pipeline" --json
```

### `venice-py api-keys delete`

Delete an API key. Prompts for confirmation before deletion.

```bash
venice-py api-keys delete <KEY_ID> [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--yes` | `-y` | `false` | Skip the confirmation prompt (for scripting / CI) |

```bash
venice-py api-keys delete key_abc123

# Non-interactive deletion (no confirmation prompt)
venice-py api-keys delete key_abc123 --yes
```

---

## Lint

`venice-py lint <path>` walks Python source files and flags v1 / OpenAI-style / non-idiomatic patterns that won't work or work sub-optimally on Venice SDK v2 — `AsyncVeniceClient` imports, hardcoded model IDs, `max_tokens=` kwargs, `PaymentRequiredError.payment_instructions` accesses, and a dozen others.

Output is in flake8-compatible `path:line:col: CODE message` format, suitable for editor integrations and CI pipelines.

### Quick examples

```bash
# Lint a directory tree
venice-py lint src/

# Lint a single file
venice-py lint path/to/script.py

# Treat informational findings (V401) as errors — exit 1 if any are present
venice-py lint --strict src/

# Filter to specific rule codes
venice-py lint --code V100 --code V200 src/
```

### Exit codes

- `0` — clean, or only informational findings (without `--strict`).
- `1` — at least one error-level finding (or any finding with `--strict`).

### Rule codes

| Code | Severity | Pattern |
|------|----------|---------|
| `V000` | error | could not parse (syntax error) |
| `V100` | error | `AsyncVeniceClient` import or call (gone in v2 — use `VeniceClient`) |
| `V101` | error | `client.image.generate(...)` (gone — use `client.image.create(...)`) |
| `V102` | error | `client.audio.generate_music(...)` (gone — use `client.music.run(...)`) |
| `V103` | error | `client.audio.generate_speech(...)` (gone — use `client.audio.create_speech(...)`) |
| `V104` | error | `client.video.queue(...)` (gone — use `client.video.run/submit(...)`) |
| `V105` | error | `client.video.complete(...)` (gone — use `client.video.cancel(...)`) |
| `V106` | error | `client.embeddings.generate(...)` (gone — use `client.embeddings.create(...)`) |
| `V107` | error | `client.audio.transcribe(audio=...)` (kwarg renamed — use `file=`) |
| `V200` | error | `max_tokens=` kwarg (removed — use `max_completion_tokens=`) |
| `V300` | error | hardcoded model string on `model=` (use `client.models.resolve_*()`) |
| `V401` | informational | `client.chat.completions.create(stream=True)` (works, but bypasses the v2 streaming idiom) |
| `V501` | error | `BudgetManager(limit=...)` (use `daily_usd=` / `monthly_usd=`) |
| `V502` | error | `tracker.total` accessor (real attr is `total_cost_usd`) |
| `V503` | error | `tracker.calls` accessor (use `len(tracker.requests)` or `total_tokens`) |
| `V601` | error | `e.payment_instructions` access (real attr is `e.body`) |

The lint runs entirely on AST — no execution of user code. Use it as a pre-commit hook or in CI to catch v1 leftovers before they reach production.

---

## Health

`venice-py health` runs a sequence of small checks against the Venice API and reports each one. Use it after configuring a fresh environment, when something feels off, or as a CI smoke test.

### Quick examples

```bash
# Default checks: API key, models catalog, billing balance
venice-py health

# Add a tiny embedding ping (small cost, typically <$0.001)
venice-py health --full

# Add an x402 prepaid-ledger balance read using a wallet env var
venice-py health --wallet                                  # default env var: VENICE_X402_TEST_PRIVATE_KEY
venice-py health --wallet --wallet-env MY_WALLET_KEY       # custom env var

# Combine
venice-py health --full --wallet
```

### Default checks

| Check | What it does |
|-------|--------------|
| `api_key` | Verifies a key is reachable via `VENICE_API_KEY` env var or saved config |
| `models.list` | Fetches the `text` model catalog (auth + connectivity sanity) |
| `billing.get_balance` | Reads the account's USD / DIEM balance |

### Optional checks (opt-in via flags)

| Flag | Adds | Notes |
|------|------|-------|
| `--full` | `embeddings.create(input="ping")` | Small cost; verifies the embedding endpoint |
| `--wallet` | `client.x402.balance(auth=X402Auth(...))` | Requires the `[x402]` extra and a wallet private key in `--wallet-env` (default: `VENICE_X402_TEST_PRIVATE_KEY`) |

### Exit codes

- `0` — every check passed.
- `1` — at least one check failed (the failing line is printed to stderr).

### Sample output

```
ℹ️  venice-py health
  ✓ api_key                  present (vn_a…fde9)
  ✓ models.list              78 text models reachable
  ✓ billing.get_balance      USD $35.71774095 / DIEM 0.7503080828059601
  ✓ embeddings.create        text-embedding-bge-m3 (3 tokens)
  ✓ x402.balance             0xecB6627A… $4.99814435

✅ All 5 checks passed.
```

The `--plain` flag substitutes plain-text equivalents for the emojis / colors.

---

## Plain Mode

The `--plain` flag disables all Rich formatting — colors, panels, emojis, and spinners — producing clean text suitable for piping, logging, and scripting.

Plain mode affects every command:

```bash
# Plain chat output
venice --plain chat start "Hello"

# Plain model listing
venice --plain models --type text

# Plain image generation
venice --plain image generate "A mountain" --output mountain
```

In plain mode:

- Status messages use `INFO:`, `ERROR:`, `SUCCESS:` prefixes instead of emojis
- Tables are rendered as aligned text columns
- Progress spinners are replaced by text status messages
- Markdown content is printed as raw text

---

## Piping & Scripting

The CLI is designed for use in scripts and pipelines. Combine `--plain`, `--json`, and stdin piping for automation.

### Piping Input via stdin

Chat, audio speak, and embeddings support piped input:

```bash
# Pipe text to chat
echo "Summarize this in one sentence" | venice-py chat start

# Pipe a file to chat
cat README.md | venice-py chat start "Summarize this document"

# Pipe text to TTS
echo "Hello world" | venice-py audio speak --output greeting.mp3

# Pipe text to embeddings
echo "Semantic search query" | venice-py embeddings --json
```

### JSON Output for Scripting

Most commands support `--json` for machine-readable output:

```bash
# Get model list as JSON
venice-py models --json | jq '.[] | .id'

# Get account balance as JSON
venice-py account balance --json | jq '.balances.usd'

# Get character list as JSON
venice-py characters list --json | jq '.[].slug'

# Get API keys as JSON
venice-py api-keys list --json

# Chat with JSON response
venice-py chat start --no-stream --json-output "What is 2+2?"
```

### Combining Plain Mode with Pipelines

```bash
# Extract just the response text
venice --plain chat start "What is the capital of France?" 2>/dev/null | grep "^Assistant:" | cut -d: -f2-

# Batch process a list of prompts
while IFS= read -r prompt; do
  venice --plain image generate "$prompt" --save-dir ./batch-output
done < prompts.txt

# Generate embeddings for multiple texts
cat sentences.txt | while IFS= read -r line; do
  venice --plain embeddings "$line" --json >> all_embeddings.jsonl
done
```

### Usage in CI/CD

```bash
# Verify API key works
venice --plain account balance || exit 1

# Generate assets in a build pipeline
venice --plain image generate "App icon" --size 512x512 --output icon --save-dir ./assets
venice --plain audio speak "Welcome to our app" --output welcome.mp3 --save-dir ./assets
```

---

## Troubleshooting

### API Key Not Found

```bash
export VENICE_API_KEY="your-key"
# or
venice-py configure
```

### Module Not Found

```bash
pip install 'venice-py[cli]'
```

### Command Not Found

```bash
# If using Poetry
poetry run venice-py --help

# Or activate the virtual environment
poetry shell
venice-py --help
```

### Model Not Found

Run `venice-py models` to see available models. Use `venice-py models --type text` or `venice-py models --type image` to filter by type. Model IDs must match exactly.

---

## Links

- [Venice AI Documentation](https://docs.venice.ai)
- [API Reference](https://docs.venice.ai/api)
- [Main SDK README](https://github.com/sethbang/venice-py/blob/main/README.md)
- [Examples](https://github.com/sethbang/venice-py/blob/main/examples/)
- [Changelog](https://github.com/sethbang/venice-py/blob/main/CHANGELOG.md)
