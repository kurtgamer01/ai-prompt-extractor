import base64
import io
import os
import json
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

def _extract_output_text(resp) -> str:
    # Prefer convenience field if present and non-empty
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    # Fall back to parsing resp.output
    out = getattr(resp, "output", None)
    texts = []

    if out:
        for item in out:
            # item may be dict-like or object-like
            if isinstance(item, dict):
                content = item.get("content")
            else:
                content = getattr(item, "content", None)

            if not content:
                continue

            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                else:
                    btype = getattr(block, "type", None)

                if btype == "output_text":
                    if isinstance(block, dict):
                        txt = block.get("text", "")
                    else:
                        txt = getattr(block, "text", "")
                    if txt:
                        texts.append(txt)

                elif btype == "refusal":
                    if isinstance(block, dict):
                        txt = block.get("refusal", "")
                    else:
                        txt = getattr(block, "refusal", "")
                    if txt:
                        texts.append(f"[REFUSAL] {txt}")

    return "\n".join(texts).strip()

SECTION_KEYS = [
    "subject_scene",
    "style",
    "clothing_accessories",
    "pose_orientation",
    "expression_gaze",
    "lighting",
    "camera_framing",
]

ALL_JSON_KEYS = SECTION_KEYS + ["negative"]

DEFAULT_SECTION_ORDER = ",".join(SECTION_KEYS)

def _empty_fields_dict() -> dict:
    return {k: "" for k in ALL_JSON_KEYS}

def _build_json_directive_7(
    include_subject_scene: bool,
    include_style: bool,
    include_clothing_accessories: bool,
    include_pose_orientation: bool,
    include_expression_gaze: bool,
    include_lighting: bool,
    include_camera_framing: bool,
) -> str:
    enabled = []
    if include_subject_scene: enabled.append("subject_scene")
    if include_style: enabled.append("style")
    if include_clothing_accessories: enabled.append("clothing_accessories")
    if include_pose_orientation: enabled.append("pose_orientation")
    if include_expression_gaze: enabled.append("expression_gaze")
    if include_lighting: enabled.append("lighting")
    if include_camera_framing: enabled.append("camera_framing")

    enabled_csv = ", ".join(enabled) if enabled else "none"

    schema = _empty_fields_dict()

    return (
        "Return ONLY valid JSON (no markdown, no code fences, no commentary).\n"
        f"JSON schema (keys must match exactly): {json.dumps(schema)}\n"
        f"Enabled sections: {enabled_csv}\n"
        "Hard rules:\n"
        "- Return ONLY the enabled section keys.\n"
        "- Do NOT include disabled section keys at all.\n"
        "- Do NOT include any information that belongs to a disabled section.\n"
        "- Do NOT infer or invent details; if unsure, omit the field.\n"
        "- Output the JSON on a single line with no extra whitespace.\n"
    )

def _safe_parse_json_object(text: str) -> dict:
    if not isinstance(text, str):
        return {}
    s = text.strip()
    if not s:
        return {}

    # direct parse
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # fallback: extract first {...}
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            obj = json.loads(s[l:r+1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {}

def _guard_and_blank_fields(
    obj: dict,
    include_subject_scene: bool,
    include_style: bool,
    include_clothing_accessories: bool,
    include_pose_orientation: bool,
    include_expression_gaze: bool,
    include_lighting: bool,
    include_camera_framing: bool,
) -> dict:
    out = _empty_fields_dict()

    if isinstance(obj, dict):
        for k in ALL_JSON_KEYS:
            v = obj.get(k, "")
            out[k] = v if isinstance(v, str) else ""

    if not include_subject_scene: out["subject_scene"] = ""
    if not include_style: out["style"] = ""
    if not include_clothing_accessories: out["clothing_accessories"] = ""
    if not include_pose_orientation: out["pose_orientation"] = ""
    if not include_expression_gaze: out["expression_gaze"] = ""
    if not include_lighting: out["lighting"] = ""
    if not include_camera_framing: out["camera_framing"] = ""

    return out


def _parse_section_order(order: str) -> list:
    allowed = set(SECTION_KEYS)
    seen = set()
    out = []
    for part in (order or "").split(","):
        k = part.strip()
        if not k:
            continue
        k = k.lower()
        if k in allowed and k not in seen:
            out.append(k)
            seen.add(k)

    return out if out else list(SECTION_KEYS)


def _combine_sections(fields: dict, order: list) -> str:
    parts = []
    for k in order:
        v = (fields.get(k, "") or "").strip()
        if v:
            parts.append(v)
    return ", ".join(parts)


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
                "max_output_tokens": ("INT", {"default": 512, "min": 32, "max": 2048}),
                "jpeg_quality": ("INT", {"default": 90, "min": 30, "max": 95}),

                # Section toggles (match SECTION_KEYS exactly)
                "include_subject_scene": ("BOOLEAN", {"default": True}),
                "include_style": ("BOOLEAN", {"default": True}),
                "include_clothing_accessories": ("BOOLEAN", {"default": True}),
                "include_pose_orientation": ("BOOLEAN", {"default": True}),
                "include_expression_gaze": ("BOOLEAN", {"default": True}),
                "include_lighting": ("BOOLEAN", {"default": True}),
                "include_camera_framing": ("BOOLEAN", {"default": True}),

                # User-controlled order (CSV, snake_case keys)
                "section_order": ("STRING", {"default": DEFAULT_SECTION_ORDER}),
            }
        }
    
    def run(
            self,
            image,
            llm_model: str,
            max_output_tokens: int,
            jpeg_quality: int,
            include_subject_scene: bool,
            include_style: bool,
            include_clothing_accessories: bool,
            include_pose_orientation: bool,
            include_expression_gaze: bool,
            include_lighting: bool,
            include_camera_framing: bool,
            section_order: str,
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
            
            instructions = instructions + "\n\n" + _build_json_directive_7(
                include_subject_scene=include_subject_scene,
                include_style=include_style,
                include_clothing_accessories=include_clothing_accessories,
                include_pose_orientation=include_pose_orientation,
                include_expression_gaze=include_expression_gaze,
                include_lighting=include_lighting,
                include_camera_framing=include_camera_framing,
            )

            response = client.responses.create(
                model=llm_model,
                instructions=instructions,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Generate an SDXL prompt from this image and return it as JSON per the instructions."},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
                text={"format": {"type": "json_object"}},
                reasoning={"effort": "low"},
                max_output_tokens=int(max_output_tokens),
            )

            try:
                print("[APIPromptExtractorSDXL] output_text:", repr(getattr(response, "output_text", None)))
                #print("[APIPromptExtractorSDXL] output:", getattr(response, "output", None))
            except Exception as _:
                pass

            raw = _extract_output_text(response)
            if not raw:
                print("[APIPromptExtractorSDXL] EMPTY TEXT OUTPUT. Full response dump follows:")
                try:
                    print(response.model_dump())
                except Exception:
                    print(response)
                return ("[APIPromptExtractorSDXL] Empty model text output (see console).",)

            obj = _safe_parse_json_object(raw)
            if not obj:
                print("[APIPromptExtractorSDXL] Failed to parse JSON. Raw output follows:")
                print(repr(raw))
                return ("[APIPromptExtractorSDXL] Invalid JSON output (see console).",)

            fields = _guard_and_blank_fields(
                obj,
                include_subject_scene=include_subject_scene,
                include_style=include_style,
                include_clothing_accessories=include_clothing_accessories,
                include_pose_orientation=include_pose_orientation,
                include_expression_gaze=include_expression_gaze,
                include_lighting=include_lighting,
                include_camera_framing=include_camera_framing,
            )

            order = _parse_section_order(section_order)
            combined = _combine_sections(fields, order)

            return (_normalize_whitespace(combined),)



        except Exception as e:
            print(f"[APIPromptExtractorSDXL] API call failed: {e}")
            return ("",)


NODE_CLASS_MAPPINGS = {
    "APIPromptExtractorSDXL": APIPromptExtractorSDXL
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIPromptExtractorSDXL": "API Prompt Extractor (SDXL)"
}

