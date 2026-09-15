"""Mask/unmask LaTeX spans so text-cleaning never mangles math notation."""
import re
import uuid

_LATEX_PATTERN = re.compile(
    r"\$\$.*?\$\$"      # $$ ... $$
    r"|\$.*?\$"          # $ ... $
    r"|\\\[.*?\\\]"       # \[ ... \]
    r"|\\\(.*?\\\)",      # \( ... \)
    re.DOTALL,
)


def mask_latex(text: str) -> tuple[str, dict[str, str]]:
    """Replace every LaTeX span with an opaque placeholder token.

    Returns the masked text plus a vault mapping placeholder -> original,
    so the spans can be restored exactly (whitespace, backslashes and all)
    after any aggressive cleaning has run.
    """
    vault: dict[str, str] = {}

    def _replace(match: re.Match) -> str:
        key = f"\u27e6LATEX_{uuid.uuid4().hex[:8]}\u27e7"
        vault[key] = match.group(0)
        return key

    return _LATEX_PATTERN.sub(_replace, text), vault


def unmask_latex(text: str, vault: dict[str, str]) -> str:
    for key, original in vault.items():
        text = text.replace(key, original)
    return text
