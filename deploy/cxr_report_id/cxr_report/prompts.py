"""
The Indonesian prompts both adapters were trained with (rrg_indo/rrg/prompts.py, without the
optional classifier hints, which these models did not use). Keep the text byte-for-byte: the
adapters learned these exact sentences, and a reworded prompt degrades the reports silently.
"""

TASKS = ("findings", "impression")  # TEMUAN, KESAN

TASK_INSTRUCTION = {
    "findings": "Tuliskan bagian TEMUAN dari laporan radiologi untuk foto toraks ini.",
    "impression": "Tuliskan bagian KESAN dari laporan radiologi untuk foto toraks ini.",
}
VIEW_TEXT = {"AP": "AP (portabel)", "PA": "PA", "LATERAL": "lateral", "LL": "lateral", "RL": "lateral"}


def normalize_view(view: str | None) -> str | None:
    """'pa' -> 'PA'; None / '' -> None (unknown view: the prompt then has no projection line,
    as for the training images without view metadata)."""
    if view is None or not str(view).strip():
        return None
    v = str(view).strip().upper()
    if v not in VIEW_TEXT:
        raise ValueError(f"unknown view {view!r}; use one of {sorted(VIEW_TEXT)} or None")
    return v


def build_prompt(task: str, view: str | None = None) -> str:
    lines = ["Anda adalah seorang ahli radiologi. " + TASK_INSTRUCTION[task]]
    view = normalize_view(view)
    if view:
        lines.append(f"Proyeksi: {VIEW_TEXT[view]}.")
    lines.append("Jawab dalam Bahasa Indonesia.")
    return "\n".join(lines)
