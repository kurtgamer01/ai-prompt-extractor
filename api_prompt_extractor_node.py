import base64
import io
import os
from typing import Tuple

import numpy as np
from PIL import Image

INSTRUCTIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    "rules",
    "sdxl_prompt_instructions.txt"
)


try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None
    _openai_import_error = e

def _load_instructions(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[APIPromptExtractorSDXL] Failed to load instructions file: {e}")
        return ""

def _comfy_image_to_jpeg_base64(image_tensor, quality: int = 90) -> str:
    """
    Convert ComfyUI IMAGE tensor [B,H,W,C] in [0,1] to base64 JPEG.
    Uses first batch item only.
    """
    import torch  # ComfyUI always has torch

    if image_tensor is None:
        raise ValueError("image_tensor is None")

    if len(image_tensor.shape) != 4 or image_tensor.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected IMAGE shape: {tuple(image_tensor.shape)}")

    img = image_tensor[0].detach().cpu().numpy()
    img = np.clip(img, 0.0, 1.0)

    if img.shape[-1] == 4:
        img = img[..., :3]

    img_u8 = (img * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(img_u8, mode="RGB")

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


class APIPromptExtractorSDXL:
    """
    IMAGE -> STRING
    Expects the API to return a valid SDXL prompt <= 75 tokens.
    """

    CATEGORY = "API Prompt Extractor"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "llm_model": ("STRING", {"default": "gpt-5-mini"}),
                "max_output_tokens": ("INT", {"default": 160, "min": 32, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0}),
                "jpeg_quality": ("INT", {"default": 90, "min": 30, "max": 95}),
            }
        }

    def run(
        self,
        image,
        llm_model: str,
        max_output_tokens: int,
        temperature: float,
        jpeg_quality: int,
    ) -> Tuple[str]:
        # Soft-fail only — never crash the graph
        if OpenAI is None:
            print(f"[APIPromptExtractorSDXL] openai import failed: {_openai_import_error}")
            return ("",)

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("[APIPromptExtractorSDXL] OPENAI_API_KEY not set")
            return ("",)

        try:
            b64 = _comfy_image_to_jpeg_base64(image, quality=jpeg_quality)
            data_url = f"data:image/jpeg;base64,{b64}"

            client = OpenAI(api_key=api_key)

            instructions = _load_instructions(INSTRUCTIONS_PATH)
            if not instructions:
                return ("",)

            response = client.responses.create(
                model=llm_model,
                instructions=instructions,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Generate an SDXL prompt from this image."},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
                max_output_tokens=int(max_output_tokens),
                temperature=float(temperature),
            )

            raw = getattr(response, "output_text", "") or ""
            return (_normalize_whitespace(raw),)

        except Exception as e:
            print(f"[APIPromptExtractorSDXL] API call failed: {e}")
            return ("",)


NODE_CLASS_MAPPINGS = {
    "APIPromptExtractorSDXL": APIPromptExtractorSDXL
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIPromptExtractorSDXL": "API Prompt Extractor (SDXL)"
}

