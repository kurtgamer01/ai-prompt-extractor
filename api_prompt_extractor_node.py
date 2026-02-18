import base64
import io
import os
import json
from typing import Tuple

import numpy as np
from PIL import Image

NODE_DIR = os.path.dirname(__file__)

# Version-controlled rules shipped with the repo
RULES_DIR = os.path.join(NODE_DIR, "rules")

# Local-only rules (gitignored)
CUSTOM_RULES_DIR = os.path.join(NODE_DIR, "custom_rules")

# Only these two folders are allowed sources
ALLOWED_RULE_DIRS = {
    "rules": RULES_DIR,
    "custom_rules": CUSTOM_RULES_DIR,
}

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None
    _openai_import_error = e

# ---- LLM provider/model registry (canonical IDs: "provider:model") ----

LLM_PROVIDERS = {
    # OpenAI
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",      # optional override
        "default_base_url": "",                 # empty = use SDK default
        "models": [
            "gpt-5-mini",
            "gpt-5",
        ],
    },
    # xAI / Grok (OpenAI-compatible API)
    "xai": {
        "env_key": "XAI_API_KEY",
        "base_url_env": "XAI_BASE_URL",         # optional override
        "default_base_url": "https://api.x.ai/v1",
        "models": [
            # Add Grok Models here
            "grok-4-1-fast-reasoning",
            "grok-4-1-fast-non-reasoning",
        ],
    },
    # Placeholder for future local routing (node can soft-fail for now)
    "local": {
        "env_key": "",
        "base_url_env": "",
        "default_base_url": "",
        "models": [
            "llava",
        ],
    },
}

BASE_SECTION_ORDER = [
    "subject",
    "scene",
    "lighting",
    "camera_framing",
    "pose_orientation",
    "expression_gaze",
    "clothing_accessories",
    "style",
]

MISC_POSITION_CHOICES = [
    "end",
    "start",
    "after_subject",
    "after_scene",
    "after_lighting",
    "after_camera_framing",
    "after_pose_orientation",
    "after_expression_gaze",
    "after_clothing_accessories",
    "after_style",
]


def _insert_misc(order: list, misc_key: str, position: str) -> list:
    pos = (position or "end").strip().lower()
    out = [k for k in order if k != misc_key]  # de-dupe

    if pos == "start":
        return [misc_key] + out
    if pos == "end":
        return out + [misc_key]

    if pos.startswith("after_"):
        anchor = pos.replace("after_", "", 1)
        if anchor in out:
            i = out.index(anchor) + 1
            return out[:i] + [misc_key] + out[i:]
        # if anchor missing, fall through to end

    return out + [misc_key]


def _build_order_with_misc(
    misc1: str,
    misc2: str,
    misc1_position: str,
    misc2_position: str,
    base_order: list = None,
) -> list:
    order = list(base_order or BASE_SECTION_ORDER)

    if (misc1 or "").strip():
        order = _insert_misc(order, "misc1", misc1_position)

    if (misc2 or "").strip():
        order = _insert_misc(order, "misc2", misc2_position)

    return order


def _llm_model_choices() -> list:
    choices = []
    for provider, cfg in LLM_PROVIDERS.items():
        for model in cfg.get("models", []):
            choices.append(f"{provider}:{model}")
    return sorted(choices)


def _parse_llm_model_id(llm_model: str) -> Tuple[str, str]:
    """
    Accepts:
      - "provider:model" (preferred)
      - "model" (legacy/back-compat; assumed openai)
    """
    s = (llm_model or "").strip()
    if ":" in s:
        provider, model = s.split(":", 1)
        return provider.strip().lower(), model.strip()
    return "openai", s


def _get_provider_cfg(provider: str) -> dict:
    return LLM_PROVIDERS.get((provider or "").strip().lower(), {})


def _load_instructions(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[APIPromptExtractorSDXL] Failed to load instructions file: {e}")
        return ""
def _list_rule_files() -> list:
    """
    Dropdown entries are repo-relative paths like:
      - rules/<file>.txt
      - custom_rules/<file>.txt
    """
    out = []

    for rel_dir, abs_dir in ALLOWED_RULE_DIRS.items():
        try:
            if not os.path.isdir(abs_dir):
                continue

            files = [
                f for f in os.listdir(abs_dir)
                if os.path.isfile(os.path.join(abs_dir, f)) and f.lower().endswith(".txt")
            ]
            files.sort()
            out.extend([f"{rel_dir}/{f}" for f in files])

        except Exception as e:
            print(f"[APIPromptExtractorSDXL] Failed to list {rel_dir} dir: {e}")

    if not out:
        return ["rules/sdxl_prompt_instructions.txt"]

    return out


def _resolve_rules_path(rules_file: str) -> str:
    """
    Resolve a dropdown value to an absolute file path, constrained to:
      - rules/
      - custom_rules/
    Prevents path traversal and rejects missing/non-txt files.
    """
    rel = (rules_file or "").strip().replace("\\", "/")
    if not rel:
        return ""

    prefix = rel.split("/", 1)[0].strip()
    if prefix not in ALLOWED_RULE_DIRS:
        print(f"[APIPromptExtractorSDXL] Invalid rules prefix: {prefix}")
        return ""

    abs_path = os.path.abspath(os.path.normpath(os.path.join(NODE_DIR, rel)))

    allowed_root = os.path.abspath(ALLOWED_RULE_DIRS[prefix])
    if not (abs_path == allowed_root or abs_path.startswith(allowed_root + os.sep)):
        print(f"[APIPromptExtractorSDXL] Invalid rules path (escape attempt): {rules_file}")
        return ""

    if not abs_path.lower().endswith(".txt"):
        print(f"[APIPromptExtractorSDXL] Rules file must be .txt: {rules_file}")
        return ""

    if not os.path.isfile(abs_path):
        print(f"[APIPromptExtractorSDXL] Rules file not found: {abs_path}")
        return ""

    return abs_path

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

def _apply_override(enabled: bool, extracted: str, override: str) -> str:
    if enabled:
        return (extracted or "").strip()
    return (override or "").strip()

SENTENCE_CHUNK_ORDER = [
    "subject",
    "scene",
    "clothing_accessories",
    "pose_orientation",
    "expression_gaze",
    "lighting",
    "camera_framing",
    "style",
    "misc1",
    "misc2",
]

Z_IMAGE_TURBO_ORDER = [
    "subject",
    "scene",
    "style",
    "lighting",
    "camera_framing",
    "pose_orientation",
    "expression_gaze",
    "clothing_accessories",
    "misc1",
    "misc2",
]

SECTION_KEYS = [
    "subject",
    "scene",
    "style",
    "clothing_accessories",
    "pose_orientation",
    "expression_gaze",
    "lighting",
    "camera_framing",
]

MISC_KEYS = ["misc1", "misc2"]

ALL_JSON_KEYS = SECTION_KEYS + ["negative"]

DEFAULT_SECTION_ORDER = ",".join(SECTION_KEYS)

def _empty_fields_dict() -> dict:
    return {k: "" for k in ALL_JSON_KEYS}

def _build_json_directive_8(
    include_subject: bool,
    include_scene: bool,
    include_style: bool,
    include_clothing_accessories: bool,
    include_pose_orientation: bool,
    include_expression_gaze: bool,
    include_lighting: bool,
    include_camera_framing: bool,
) -> str:
    enabled = []
    if include_subject: enabled.append("subject")
    if include_scene: enabled.append("scene")
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

def _combine_sections_sentence(fields: dict, order: list) -> str:
    parts = []
    for k in order:
        v = (fields.get(k, "") or "").strip()
        if not v:
            continue
        # ensure each chunk ends with a period for readability
        if v[-1] not in ".!?":
            v = v + "."
        parts.append(v)
    return " ".join(parts).strip()

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
    include_subject: bool,
    include_scene: bool,
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

    if not include_subject: out["subject"] = ""
    if not include_scene: out["scene"] = ""
    if not include_style: out["style"] = ""
    if not include_clothing_accessories: out["clothing_accessories"] = ""
    if not include_pose_orientation: out["pose_orientation"] = ""
    if not include_expression_gaze: out["expression_gaze"] = ""
    if not include_lighting: out["lighting"] = ""
    if not include_camera_framing: out["camera_framing"] = ""

    return out


def _parse_section_order(order: str) -> list:
    allowed = set(SECTION_KEYS) | set(MISC_KEYS)
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


def _combine_sections(fields: dict, order: list, misc1: str, misc2: str) -> str:
    misc_map = {
        "misc1": (misc1 or "").strip(),
        "misc2": (misc2 or "").strip(),
    }

    parts = []
    sentence_like = False

    for k in order:
        if k in misc_map:
            v = misc_map[k]
        else:
            v = (fields.get(k, "") or "").strip()

        if not v:
            continue

        v = v.strip()

        # detect sentence-style chunks
        if v.endswith((".", "!", "?")):
            sentence_like = True

        parts.append(v)

    if not parts:
        return ""

    if sentence_like:
        # sentence grammar → space join
        cleaned = [p.rstrip(" ,") for p in parts]
        return " ".join(cleaned)

    # tag grammar → comma join
    cleaned = [p.rstrip(" .") for p in parts]
    return ", ".join(cleaned)

FALLBACK_XAI_MODEL = "grok-4-1-fast-non-reasoning"

def _is_refusal_like(raw: str) -> bool:
    if not isinstance(raw, str):
        return True
    s = raw.strip()
    if not s:
        return True

    # If _extract_output_text captured a refusal block
    if "[REFUSAL]" in s:
        return True

    # Plain-text refusal phrases sometimes appear
    lowered = s.lower()
    if "i can’t assist" in lowered or "i can't assist" in lowered:
        return True

    # JSON refusal object pattern
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and ("error" in obj or "refusal" in obj):
            return True
    except Exception:
        pass

    return False

def _make_client_for_provider(provider: str) -> "OpenAI":
    cfg = _get_provider_cfg(provider)
    if not cfg:
        raise RuntimeError(f"UNKNOWN_PROVIDER:{provider}")

    env_key = (cfg.get("env_key") or "").strip()
    api_key = os.getenv(env_key, "").strip() if env_key else ""
    if not api_key:
        raise RuntimeError(f"MISSING_API_KEY:{env_key}")

    base_url_env = (cfg.get("base_url_env") or "").strip()
    base_url = (os.getenv(base_url_env, "").strip() if base_url_env else "") or (cfg.get("default_base_url") or "").strip()

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)

def _call_and_get_raw(client, req_kwargs: dict) -> str:
    resp = client.responses.create(**req_kwargs)
    return _extract_output_text(resp) or ""

def _call_parse_or_raise(client, req_kwargs: dict) -> dict:
    raw = _call_and_get_raw(client, req_kwargs)

    if _is_refusal_like(raw):
        raise RuntimeError("MODEL_REFUSAL_OR_EMPTY")

    obj = _safe_parse_json_object(raw)
    if not obj:
        raise RuntimeError("INVALID_JSON")

    # Parsed but is actually refusal object
    if isinstance(obj, dict) and ("error" in obj or "refusal" in obj):
        raise RuntimeError("REFUSAL_OBJECT")

    return obj


class APIPromptExtractorSDXL:
    """
    IMAGE -> STRING
    Expects the API to return a valid SDXL prompt <= 75 tokens.
    """

    CATEGORY = "API Prompt Extractor"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "llm_model": (_llm_model_choices(), {"default": "openai:gpt-5-mini"}),
                "rules_file": (_list_rule_files(), {"default": "rules/sdxl_prompt_instructions.txt"}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 32, "max": 5096}),
                "jpeg_quality": ("INT", {"default": 90, "min": 30, "max": 95}),
                "combine_mode": (["section_order", "simplified_juggernaut", "sentence_chunks","z-image-turbo",], {"default": "section_order"}),
                # Section toggles (match SECTION_KEYS exactly)
                "include_subject": ("BOOLEAN", {"default": True}),
                "subject_override": ("STRING", {"default": "", "multiline": True}),

                "include_scene": ("BOOLEAN", {"default": True}),
                "scene_override": ("STRING", {"default": "", "multiline": True}),


                "include_style": ("BOOLEAN", {"default": True}),
                "style_override": ("STRING", {"default": "", "multiline": True}),

                "include_clothing_accessories": ("BOOLEAN", {"default": True}),
                "clothing_accessories_override": ("STRING", {"default": "", "multiline": True}),
                "include_pose_orientation": ("BOOLEAN", {"default": True}),
                "pose_orientation_override": ("STRING", {"default": "", "multiline": True}),
                "include_expression_gaze": ("BOOLEAN", {"default": True}),
                "expression_gaze_override": ("STRING", {"default": "", "multiline": True}),
                "include_lighting": ("BOOLEAN", {"default": True}),
                "lighting_override": ("STRING", {"default": "", "multiline": True}),
                "include_camera_framing": ("BOOLEAN", {"default": True}),
                "camera_framing_override": ("STRING", {"default": "", "multiline": True}),
                
                "misc1": ("STRING", {"default": "", "multiline": True}),
                "misc2": ("STRING", {"default": "", "multiline": True}),
                "misc1_position": (MISC_POSITION_CHOICES, {"default": "end"}),
                "misc2_position": (MISC_POSITION_CHOICES, {"default": "end"}),
            }
        }
    
    def run(
        self,
        image,
        llm_model: str,
        rules_file: str,
        max_output_tokens: int,
        jpeg_quality: int,
        combine_mode: str,

        include_subject: bool,
        subject_override: str,

        include_style: bool,
        style_override: str,

        include_scene: bool,
        scene_override: str,

        include_clothing_accessories: bool,
        clothing_accessories_override: str,

        include_pose_orientation: bool,
        pose_orientation_override: str,

        include_expression_gaze: bool,
        expression_gaze_override: str,

        include_lighting: bool,
        lighting_override: str,  

        include_camera_framing: bool,
        camera_framing_override: str,

        misc1: str,
        misc2: str,
        misc1_position: str,
        misc2_position: str,

                
    ) -> Tuple[str, str]:
        # Soft-fail only — never crash the graph
        if OpenAI is None:
            print(f"[APIPromptExtractorSDXL] openai import failed: {_openai_import_error}")
            return ("", "")

        provider, model_name = _parse_llm_model_id(llm_model)
        cfg = _get_provider_cfg(provider)
        if not cfg:
            print(f"[APIPromptExtractorSDXL] Unknown provider: {provider}")
            return ("", "")

        # If you later implement local routing, replace this soft-fail
        if provider == "local":
            print("[APIPromptExtractorSDXL] provider=local not implemented in this node yet")
            return ("", "")

        env_key = (cfg.get("env_key") or "").strip()
        api_key = os.getenv(env_key, "").strip() if env_key else ""
        if not api_key:
            print(f"[APIPromptExtractorSDXL] {env_key} not set for provider={provider}")
            return ("", "")

        # Optional base_url override per provider (lets you swap endpoints without code changes)
        base_url_env = (cfg.get("base_url_env") or "").strip()
        base_url = (os.getenv(base_url_env, "").strip() if base_url_env else "") or (cfg.get("default_base_url") or "").strip()

        try:
            b64 = _comfy_image_to_jpeg_base64(image, quality=jpeg_quality)
            data_url = f"data:image/jpeg;base64,{b64}"

            # OpenAI SDK supports base_url; if empty, use default
            if base_url:
                client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                client = OpenAI(api_key=api_key)


            rules_path = _resolve_rules_path(rules_file)
            if not rules_path:
                return ("", "")

            instructions = _load_instructions(rules_path)
            if not instructions:
                return ("", "")
            
            instructions = instructions + "\n\n" + _build_json_directive_8(
                include_subject=include_subject,
                include_scene=include_scene,
                include_style=include_style,
                include_clothing_accessories=include_clothing_accessories,
                include_pose_orientation=include_pose_orientation,
                include_expression_gaze=include_expression_gaze,
                include_lighting=include_lighting,
                include_camera_framing=include_camera_framing,
            )

            req_kwargs = dict(
                model=model_name,
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
                max_output_tokens=int(max_output_tokens),
            )

            # OpenAI supports this; Grok (xAI) is rejecting it
            if provider == "openai":
                req_kwargs["reasoning"] = {"effort": "low"}

            def build_req_kwargs(for_provider: str, for_model: str) -> dict:
                rk = dict(req_kwargs)
                rk["model"] = for_model

                # Only OpenAI gets reasoning
                if for_provider != "openai":
                    rk.pop("reasoning", None)

                return rk

            # 1) Try primary once
            try:
                obj = _call_parse_or_raise(client, build_req_kwargs(provider, model_name))
            except Exception as e_primary:
                print(f"[APIPromptExtractorSDXL] Primary provider failed/refused: {e_primary}")

                # 2) Immediate fallback to xAI Grok
                try:
                    xclient = _make_client_for_provider("xai")
                    obj = _call_parse_or_raise(
                        xclient,
                        build_req_kwargs("xai", FALLBACK_XAI_MODEL)
                    )
                    print("[APIPromptExtractorSDXL] Fallback to xAI succeeded.")
                except Exception as e_fallback:
                    print(f"[APIPromptExtractorSDXL] Fallback to xAI failed: {e_fallback}")
                    return ("[APIPromptExtractorSDXL] Extraction failed after fallback (see console).", "")
           

            fields = _guard_and_blank_fields(
                obj,
                include_subject=include_subject,
                include_scene=include_scene,
                include_style=include_style,
                include_clothing_accessories=include_clothing_accessories,
                include_pose_orientation=include_pose_orientation,
                include_expression_gaze=include_expression_gaze,
                include_lighting=include_lighting,
                include_camera_framing=include_camera_framing,
            )

            # Apply per-section overrides when toggles are OFF
            fields["subject"] = _apply_override(include_subject, fields["subject"], subject_override)
            fields["scene"] = _apply_override(include_scene, fields["scene"], scene_override)
            fields["style"] = _apply_override(include_style, fields["style"], style_override)
            fields["clothing_accessories"] = _apply_override(include_clothing_accessories, fields["clothing_accessories"], clothing_accessories_override)
            fields["pose_orientation"] = _apply_override(include_pose_orientation, fields["pose_orientation"], pose_orientation_override)
            fields["expression_gaze"] = _apply_override(include_expression_gaze, fields["expression_gaze"], expression_gaze_override)
            fields["lighting"] = _apply_override(include_lighting, fields["lighting"], lighting_override)
            fields["camera_framing"] = _apply_override(include_camera_framing, fields["camera_framing"], camera_framing_override)

            mode = (combine_mode or "").strip().lower()

            # pick the base order per mode (WITHOUT assuming misc is at end)
            if mode == "sentence_chunks":
                base = SENTENCE_CHUNK_ORDER
            elif mode == "z-image-turbo":
                base = Z_IMAGE_TURBO_ORDER
            elif mode == "simplified_juggernaut":
                base = [
                    "subject",
                    "scene",
                    "lighting",
                    "camera_framing",
                    "pose_orientation",
                    "expression_gaze",
                    "clothing_accessories",
                    "style",
                ]
            else:
                base = BASE_SECTION_ORDER

            # build an order that respects misc placement for the chosen base
            order = _build_order_with_misc(
                misc1=misc1,
                misc2=misc2,
                misc1_position=misc1_position,
                misc2_position=misc2_position,
                base_order=base,
            )

            if mode == "sentence_chunks":
                fields["misc1"] = (misc1 or "").strip()
                fields["misc2"] = (misc2 or "").strip()
                combined = _combine_sections_sentence(fields, order)
            else:
                combined = _combine_sections(fields, order, misc1, misc2)

              
            negative = _normalize_whitespace(fields.get("negative", ""))
            return (_normalize_whitespace(combined), negative)



        except Exception as e:
            print(f"[APIPromptExtractorSDXL] API call failed: {e}")
            return ("", "")


NODE_CLASS_MAPPINGS = {
    "APIPromptExtractorSDXL": APIPromptExtractorSDXL
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIPromptExtractorSDXL": "API Prompt Extractor (SDXL)"
}

