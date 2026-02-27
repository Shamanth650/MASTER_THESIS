"""
RAG2/llm_client.py

IMPROVED VERSION:
- Better Claude response handling
- More robust JSON extraction
- Clearer error messages
- Support for Claude's thinking process
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from .config import OPENAI_MODEL


# -----------------------------
# JSON extraction (robust)
# -----------------------------

def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the first JSON object found in model output.
    
    IMPROVED: Handles Claude's tendency to add explanations before/after JSON.
    
    Strategy:
    1) Try full-string json.loads
    2) Remove markdown code fences if present
    3) Look for JSON object markers and extract
    4) Brace-balanced scan for the first {...} that parses as JSON
    """
    if not text:
        return None

    # 1) Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) Strip markdown code fences (Claude often adds these)
    cleaned = text
    if "```json" in text.lower() or "```" in text:
        # Remove ```json ... ``` or ``` ... ```
        cleaned = re.sub(r'```(?:json)?\s*', '', text)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        
        try:
            obj = json.loads(cleaned.strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) Look for JSON markers (Claude sometimes adds "Here's the JSON:" etc.)
    json_markers = [
        r'(?:here(?:\'?s| is) the json:?\s*)({\s*")',
        r'(?:json output:?\s*)({\s*")',
        r'(?:result:?\s*)({\s*")',
    ]
    
    for pattern in json_markers:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            # Start from the opening brace
            start_pos = match.start(1)
            text_from_brace = text[start_pos:]
            
            # Try to parse from this point
            try:
                obj = json.loads(text_from_brace)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    # 4) Brace-balanced scan (original logic, improved)
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        
        for i in range(start, len(text)):
            ch = text[i]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start : i + 1]
                    try:
                        obj = json.loads(blob)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break  # this candidate failed; try next '{'
        
        start = text.find("{", start + 1)

    return None


# -----------------------------
# OpenAI call
# -----------------------------

def _openai_call(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float | None = None,
    seed: int | None = None,
    max_output_tokens: int | None = None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Put it in .env at repo root.")

    if temperature is None:
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

    if seed is None:
        seed_str = os.getenv("OPENAI_SEED", "").strip()
        seed = int(seed_str) if seed_str.isdigit() else None

    client = OpenAI(api_key=api_key)

    kwargs: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    if seed is not None:
        kwargs["seed"] = seed
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens

    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


# -----------------------------
# Claude call (IMPROVED)
# -----------------------------

def _claude_call(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """
    IMPROVED Claude calling with better error handling and configuration.
    
    Env:
      - ANTHROPIC_API_KEY or CLAUDE_API_KEY (required)
      - ANTHROPIC_MODEL or CLAUDE_MODEL (optional; default: claude-sonnet-4-20250514)
      - ANTHROPIC_TEMPERATURE or CLAUDE_TEMPERATURE (optional; default: 0.0)
      - ANTHROPIC_MAX_TOKENS or CLAUDE_MAX_TOKENS (optional; default: 4096)
    """
    api_key = (
        os.getenv("ANTHROPIC_API_KEY") or 
        os.getenv("CLAUDE_API_KEY") or
        os.getenv("ANTHROPIC_KEY")
    )
    
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY missing. Set it in your .env file.\n"
            "Also accepts: CLAUDE_API_KEY or ANTHROPIC_KEY"
        )

    # Model selection with better defaults
    model = (
        os.getenv("ANTHROPIC_MODEL") or 
        os.getenv("CLAUDE_MODEL") or
        "claude-sonnet-4-20250514"  # Latest Sonnet 4
    )

    if temperature is None:
        temp_str = os.getenv("ANTHROPIC_TEMPERATURE") or os.getenv("CLAUDE_TEMPERATURE", "0.0")
        temperature = float(temp_str)

    if max_output_tokens is None:
        max_str = (
            os.getenv("ANTHROPIC_MAX_TOKENS") or 
            os.getenv("CLAUDE_MAX_TOKENS", "")
        ).strip()
        max_output_tokens = int(max_str) if max_str.isdigit() else 4096

    # --- Try anthropic SDK first ---
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        
        msg = client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text from response blocks
        parts: list[str] = []
        for block in getattr(msg, "content", []) or []:
            txt = getattr(block, "text", None)
            if isinstance(txt, str):
                parts.append(txt)
        
        result = "\n".join(parts).strip()
        
        if not result:
            raise RuntimeError("Claude returned empty response")
        
        return result

    except ModuleNotFoundError:
        # SDK not installed, fall back to HTTP
        pass
    except Exception as e:
        # If SDK exists but errors, try HTTP fallback
        import traceback
        print(f"Warning: Claude SDK error: {e}")
        print(traceback.format_exc())
        print("Attempting HTTP fallback...")

    # --- Raw HTTPS fallback ---
    try:
        import requests
    except ModuleNotFoundError:
        raise RuntimeError(
            "Neither 'anthropic' nor 'requests' library is installed.\n"
            "Install with: pip install anthropic"
        )

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.Timeout:
        raise RuntimeError("Claude API request timed out after 120 seconds")
    except Exception as e:
        raise RuntimeError(f"Claude API request failed: {e}")

    if r.status_code >= 400:
        error_detail = r.text[:1000]
        raise RuntimeError(
            f"Claude HTTP error {r.status_code}:\n{error_detail}\n\n"
            f"Check your ANTHROPIC_API_KEY and model name ({model})"
        )

    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"Failed to parse Claude response as JSON: {e}\nResponse: {r.text[:500]}")

    # Extract content
    parts = []
    for block in data.get("content", []) or []:
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    
    result = "\n".join(parts).strip()
    
    if not result:
        raise RuntimeError(f"Claude returned empty response. Full data: {data}")
    
    return result


# -----------------------------
# Public API (IMPROVED)
# -----------------------------

def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    provider: str = "openai",
    retries: int = 3,
) -> Dict[str, Any]:
    """
    IMPROVED: Better retry logic and error messages.
    
    High-level helper:
    - Calls selected provider (OpenAI or Claude)
    - Extracts JSON with robust parsing
    - Retries if JSON is malformed
    - Provides clear error messages

    Args:
      provider: "openai" | "claude"
      retries: number of retry attempts (default 3, was 2)
    
    Returns:
      Parsed JSON dict
    
    Raises:
      ValueError: if provider is invalid or JSON extraction fails after retries
      RuntimeError: if API call fails
    """
    provider_norm = (provider or "").strip().lower()
    if provider_norm not in ("openai", "claude"):
        raise ValueError(
            f"Unsupported provider: {provider}. Use 'openai' or 'claude'.\n"
            f"Set provider in UI or pass via orchestrator."
        )

    last_text = ""
    prompt = user_prompt
    attempt = 0

    for attempt in range(max(1, retries + 1)):
        try:
            # Call the appropriate provider
            if provider_norm == "openai":
                last_text = _openai_call(system_prompt, prompt)
            else:
                last_text = _claude_call(system_prompt, prompt)
            
            # Try to extract JSON
            obj = _extract_first_json_object(last_text)
            
            if obj is not None and isinstance(obj, dict):
                # Success!
                return obj
            
            # JSON extraction failed - prepare retry
            if attempt < retries:
                # Add stronger instruction for next attempt
                prompt = (
                    user_prompt
                    + "\n\n"
                    + "="*50 + "\n"
                    + "CRITICAL: Previous response was not valid JSON.\n"
                    + "You MUST respond with ONLY a single JSON object.\n"
                    + "NO explanations. NO markdown fences. NO extra text.\n"
                    + "Start your response with { and end with }\n"
                    + "="*50
                )
            
        except Exception as e:
            if attempt < retries:
                print(f"Warning: {provider_norm} call failed (attempt {attempt + 1}/{retries + 1}): {e}")
                print("Retrying...")
                continue
            else:
                raise RuntimeError(
                    f"{provider_norm} API call failed after {retries + 1} attempts: {e}"
                )

    # All retries exhausted
    raise ValueError(
        f"{provider_norm.upper()} did not return valid JSON after {retries + 1} attempts.\n"
        f"Last output (first 1000 chars):\n"
        f"{'-'*50}\n"
        f"{last_text[:1000]}\n"
        f"{'-'*50}\n\n"
        f"This usually means:\n"
        f"1. The prompt is too complex or contradictory\n"
        f"2. The model is adding explanations instead of pure JSON\n"
        f"3. The response is being truncated\n\n"
        f"Try:\n"
        f"- Simplifying the scenario\n"
        f"- Checking ANTHROPIC_MAX_TOKENS (currently: {os.getenv('ANTHROPIC_MAX_TOKENS', '4096')})\n"
        f"- Reviewing the prompts in python_prompts.py and xosc_prompts.py"
    )