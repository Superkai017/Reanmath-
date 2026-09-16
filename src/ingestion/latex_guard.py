"""Shield LaTeX from text processing by swapping formulas for opaque tokens.

Every formula is replaced by a token like ``⟦MATH_<uuid4 hex>⟧``. The token
contains no whitespace, sentence punctuation or zero-width characters, so
normalisation, Khmer segmentation and the chunker can never cut through it.
``unmask_latex`` restores the original bytes exactly.
"""
from __future__ import annotations

import re
import uuid

TOKEN_OPEN = "⟦"
TOKEN_CLOSE = "⟧"
PLACEHOLDER_PATTERN = re.compile(TOKEN_OPEN + r"MATH_[0-9a-f]{32}" + TOKEN_CLOSE)

_MATH_ENVIRONMENTS = (
    "equation", "align", "alignat", "gather", "multline", "flalign", "eqnarray",
    "displaymath", "math", "cases", "array",
    "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix",
)
_ENV_ALTERNATION = "|".join(re.escape(env) + r"\*?" for env in _MATH_ENVIRONMENTS)

# Order matters: the longest delimiters are tried first so `$$` is never read
# as two empty inline formulas.
LATEX_PATTERN = re.compile(
    # $$ ... $$ (display)
    r"(?<!\\)\$\$(?:\\.|[^$\\]|\$(?!\$))+?\$\$"
    # \[ ... \] (display)
    r"|\\\[.+?\\\]"
    # \begin{env} ... \end{env}
    r"|\\begin\{(?P<env>" + _ENV_ALTERNATION + r")\}.*?\\end\{(?P=env)\}"
    # \( ... \) (inline)
    r"|\\\(.+?\\\)"
    # $ ... $ (inline). Escaped characters are consumed as pairs, so `\$` never
    # closes a formula; a closing `$` followed by a digit is treated as currency.
    r"|(?<![\\$])\$(?!\$)(?:\\.|[^$\\\n])+?\$(?!\d)",
    re.DOTALL,
)


class LatexRestoreError(KeyError):
    """Raised when a placeholder has no entry in the vault (strict mode)."""


def _new_token() -> str:
    return f"{TOKEN_OPEN}MATH_{uuid.uuid4().hex}{TOKEN_CLOSE}"


def mask_latex(text: str) -> tuple[str, dict[str, str]]:
    """Replace each LaTeX formula with a unique token.

    Returns the masked text and a vault mapping token -> original formula.
    """
    vault: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        token = _new_token()
        vault[token] = match.group(0)
        return token

    return LATEX_PATTERN.sub(_replace, text), vault


def unmask_latex(text: str, vault: dict[str, str], *, strict: bool = False) -> str:
    """Put the original formulas back in place of their tokens.

    Tokens missing from the vault are left untouched unless ``strict`` is set,
    in which case ``LatexRestoreError`` is raised.
    """
    if not vault and not strict:
        return text

    def _restore(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in vault:
            return vault[token]
        if strict:
            raise LatexRestoreError(token)
        return token

    return PLACEHOLDER_PATTERN.sub(_restore, text)


def find_placeholders(text: str) -> list[str]:
    return PLACEHOLDER_PATTERN.findall(text)


def contains_placeholder(text: str) -> bool:
    return PLACEHOLDER_PATTERN.search(text) is not None


def sub_vault(text: str, vault: dict[str, str]) -> dict[str, str]:
    """The subset of ``vault`` whose tokens occur in ``text``."""
    return {token: vault[token] for token in dict.fromkeys(find_placeholders(text)) if token in vault}


def extract_formulas(text: str) -> list[str]:
    """All LaTeX formulas in ``text``, in order of appearance."""
    return [match.group(0) for match in LATEX_PATTERN.finditer(text)]


def strip_delimiters(formula: str) -> str:
    """The body of a formula without its surrounding math delimiters."""
    if formula.startswith("$$") and formula.endswith("$$"):
        return formula[2:-2]
    if formula.startswith(("\\[", "\\(")):
        return formula[2:-2]
    if formula.startswith("$") and formula.endswith("$"):
        return formula[1:-1]
    return formula


def is_display_formula(formula: str) -> bool:
    return formula.startswith(("$$", "\\[", "\\begin"))


def has_balanced_braces(formula: str) -> bool:
    """True when ``{``/``}`` are balanced, ignoring escaped braces."""
    depth = 0
    index = 0
    while index < len(formula):
        char = formula[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
        index += 1
    return depth == 0
