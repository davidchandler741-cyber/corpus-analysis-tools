import re
from typing import Optional

def clean_text(raw_text: str, steps: Optional[dict] = None) -> str:

    defaults = {
        "lowercase": True,
        "remove_punct": True,
        "keep_apostrophes": False,
        "keep_hyphens": False,
        "normalize_whitespace": True,
        "preserve_newlines": True,
    }

    if steps is None:
        user_steps = {}
    else:
        user_steps = steps

    unknown_keys = set(user_steps) - set(defaults)
    if unknown_keys:
        print(f"Warning: unknown step keys ignored: {unknown_keys}")

    steps = {**defaults, **user_steps}

    for k, default_val in defaults.items():
        if not isinstance(steps.get(k), bool):
            print(f"Warning: step '{k}' should be a bool; using default {default_val}.")
            steps[k] = default_val

    cleaner_text = raw_text
    if steps["lowercase"]:
        cleaner_text = cleaner_text.lower()
    if steps["remove_punct"]:
        allowed_chars = ""
        if steps["keep_apostrophes"]:
            allowed_chars = allowed_chars + "'"
        if steps["keep_hyphens"]:
            allowed_chars = allowed_chars + "-"
        regex = fr'[^\w\s{allowed_chars}]+'
        cleaner_text = re.sub(regex, "", cleaner_text)
    if steps["normalize_whitespace"]:
        if steps["preserve_newlines"]:
            lines = cleaner_text.split("\n")
            clean_lines = []
            for line in lines:
                clean_line = re.sub(r"\s+", " ", line).strip()
                clean_lines.append(clean_line)
            cleaner_text =  "\n".join(clean_lines)
        else:
            tokens = cleaner_text.split()
            cleaner_text = " ".join(tokens)
    return cleaner_text