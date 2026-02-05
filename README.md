# AI Prompt Extractor (ComfyUI Custom Node)

A ComfyUI custom node that analyzes an input image using the OpenAI Responses API and generates a **single, well-structured Stable Diffusion XL (SDXL) positive prompt line** suitable for direct use in SDXL workflows.

Repository: https://github.com/kurtgamer01/ai-prompt-extractor

---

## Features

- Image → SDXL prompt extraction
- **Selectable OpenAI models per node** (recommended default: `gpt-5-mini`)
- Enforces strict section ordering:
  - subject and scene
  - style
  - clothing
  - pose and body orientation
  - facial expression and gaze
  - lighting
  - camera and shot framing
- Outputs one comma-separated positive prompt line
- No negative prompts
- No explanations, labels, or markdown in output
- Handles multi-person scenes and clothing vs. pose constraints
- Designed for ComfyUI-native workflows (Preview Text, downstream prompt nodes)

---

## Output Format

The generated prompt always follows this exact structure:

```text
subject and scene, style, clothing, pose, expression, lighting, camera
```

Example output:

```text
two young people on a rocky beach at sunset, cinematic lifestyle photography, shearling jacket and plaid scarf camera strap, standing close shoulder to shoulder holding a lit sparkler, calm neutral expressions looking at camera, golden backlight warm rim light soft ambient glow, 50mm shallow depth medium waist shot eye-level
```

---

## Installation

Clone the repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kurtgamer01/ai-prompt-extractor.git
```

Install dependencies into your **ComfyUI virtual environment**:

```bash
pip install -r requirements.txt
```

Restart ComfyUI after installation.

---

## Requirements

Declared in `requirements.txt`:

- `openai` (Responses API client)
- `numpy`
- `Pillow`

Torch, CUDA, and ComfyUI core dependencies are intentionally **not** included and are expected to already exist in your ComfyUI environment.

---

## Model Selection

The OpenAI model is selectable per node and passed directly to the Responses API.

**Recommended default:** `gpt-5-mini`

This model provides the best balance of instruction adherence, visual understanding, speed, and cost efficiency for SDXL prompt extraction. Larger models may offer marginal improvements on complex or ambiguous scenes.

---

## Configuration

The node exposes configurable inputs inside ComfyUI, including:

- OpenAI model selection
- Maximum output tokens
- Instruction rules file
- Optional image resolution clamping (for cost control)

Behavior notes:

- Custom nodes are loaded at ComfyUI startup; a restart is required after code changes.
- The node forces text output and limits internal reasoning to avoid empty responses.
- Errors are handled gracefully and will not crash a ComfyUI graph.

---

## Intended Use Cases

- Reverse-engineering SDXL prompts from reference images
- Generating structured prompts from photos or concept art
- Building automated prompt pipelines inside ComfyUI
- Producing consistent prompts for iteration or LoRA workflows

---

## License

Apache-2.0
