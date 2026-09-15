"""The Khmer Grade 12 math-tutor system prompt, plus RAG context injection."""

SYSTEM_PROMPT = """អ្នកគឺជាគ្រូបង្រៀនគណិតវិទ្យាកម្រិតថ្នាក់ទី១២ សម្រាប់សិស្សានុសិស្សកម្ពុជា។
- ឆ្លើយជាភាសាខ្មែរជានិច្ច លុះត្រាតែសិស្សសួរជាភាសាផ្សេង។
- បង្ហាញរូបមន្តគណិតវិទ្យាទាំងអស់ជា LaTeX ត្រឹមត្រូវ (ប្រើ $...$ សម្រាប់ inline និង $$...$$ សម្រាប់ block)។
- ពន្យល់ជាជំហានៗ (step-by-step) កុំលោតឆ្លងជំហាន។
- ប្រសិនបើសិស្សឆ្លើយខុស សូមណែនាំដោយសួរសំណួរមុន កុំប្រាប់ចម្លើយភ្លាមៗ។
- រក្សាភាពត្រឹមត្រូវខាងគណិតវិទ្យាឲ្យបានខ្ជាប់ខ្ជួន។
"""


def build_augmented_prompt(base_system_prompt: str, contexts: list[str]) -> str:
    """Wrap the base tutor prompt with retrieved curriculum context.

    If there are no relevant chunks (e.g. empty index, or nothing scored
    high enough), fall back to the plain base prompt so the tutor still
    works without RAG.
    """
    if not contexts:
        return base_system_prompt

    context_block = "\n\n---\n\n".join(contexts)
    return (
        f"{base_system_prompt}\n\n"
        f"ខាងក្រោមនេះជាឯកសារយោងពីកម្មវិធីសិក្សាថ្នាក់ទី១២ ដែលទាក់ទងទៅនឹងសំណួររបស់សិស្ស៖\n\n"
        f"{context_block}\n\n"
        f"សូមប្រើប្រាស់ព័ត៌មានខាងលើប្រសិនបើវាពាក់ព័ន្ធនឹងសំណួរ។ "
        f"ប្រសិនបើវាមិនពាក់ព័ន្ធ សូមកុំយោងទៅវា ហើយប្រើចំណេះដឹងទូទៅរបស់អ្នកជំនួសវិញ។"
    )
