"""Step 82: create *separate* SARTrack v3 dashboard/pipeline variants.

Run once from project root:
    python SARTrack_Step82_Create_V3_Dashboard.py

Uses only Python standard library. Keeps existing app.py, sartrack_pipeline.py,
model checkpoints, saved v2 demo, and outputs/ untouched.
No PyTorch import, no GEE access, no training, no inference is performed.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"
PIPELINE = ROOT / "sartrack_pipeline.py"
WEIGHTS = ROOT / "models" / "best_unet_v3.pth"
NEW_APP = ROOT / "app_v3.py"
NEW_PIPELINE = ROOT / "sartrack_pipeline_v3.py"


def replace_exact_once(text: str, old: str, new: str, description: str) -> str:
    occurrences = text.count(old)
    if occurrences != 1:
        raise RuntimeError(
            f"Cannot safely patch {description}: expected one exact match, got {occurrences}. "
            "Original files were not modified."
        )
    return text.replace(old, new, 1)


def main() -> None:
    for path in (APP, PIPELINE, WEIGHTS):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required project input missing or empty: {path}")

    # Refuse overwriting even previous v3 edits. Running twice is non-destructive.
    for path in (NEW_APP, NEW_PIPELINE):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    app_text = APP.read_text(encoding="utf-8-sig")
    pipe_text = PIPELINE.read_text(encoding="utf-8-sig")

    # Model architecture and input normalization are inherited unchanged.
    # Isolate outputs because the original path contains v2 prediction maps.
    pipe_changes = [
        (
            'WEIGHTS = ROOT / "models" / "best_unet_v2.pth"',
            'WEIGHTS = ROOT / "models" / "best_unet_v3.pth"',
            "v3 checkpoint path",
        ),
        (
            'folder = ROOT / "outputs" / "live" / f"lat{latitude:+.4f}_lon{longitude:+.4f}_{start_year}_{start_month:02d}"',
            'folder = ROOT / "outputs" / "live_v3" / f"lat{latitude:+.4f}_lon{longitude:+.4f}_{start_year}_{start_month:02d}"',
            "separate v3 results directory",
        ),
        (
            'notify(f"U-Net v2 loaded on {device}; common valid pixels: {valid_common.sum():,}")',
            'notify(f"U-Net v3 loaded on {device}; common valid pixels: {valid_common.sum():,}")',
            "correct model version in logs",
        ),
    ]
    for old, new, description in pipe_changes:
        pipe_text = replace_exact_once(pipe_text, old, new, description)

    app_changes = [
        (
            'from sartrack_pipeline import run_pipeline',
            'from sartrack_pipeline_v3 import run_pipeline',
            "load v3 pipeline for live runs",
        ),
        (
            '    st.markdown("**Model:** U-Net v2")',
            '    st.markdown("**Live model:** U-Net v3")\n'
            '    st.caption("Saved Esch demo: earlier U-Net v2 predictions.")',
            "accurate sidebar provenance",
        ),
        (
            '    st.subheader("U-Net v2 — Model Information")',
            '    st.subheader("U-Net v2 — Saved Demo" if source == "demo" else "U-Net v3 — Live Inference")',
            "version-specific model details heading",
        ),
        (
            '    a.metric("Test Accuracy", "79.90%")',
            '    a.metric("Test Accuracy", "79.90%" if source == "demo" else "83.71%")',
            "correct independent test overall accuracy",
        ),
        (
            '    b.metric("Test mIoU", "38.72%")',
            '    b.metric("Test mIoU", "38.72%" if source == "demo" else "46.37%")',
            "correct independent test mIoU",
        ),
        (
            '    c.metric("Test Macro F1", "49.45%")',
            '    c.metric("Test Macro F1", "49.45%" if source == "demo" else "59.33%")',
            "correct independent test Macro F1",
        ),
    ]
    for old, new, description in app_changes:
        app_text = replace_exact_once(app_text, old, new, description)

    # Catch syntax/indentation mistakes before writing anything.
    ast.parse(pipe_text, filename=str(NEW_PIPELINE))
    ast.parse(app_text, filename=str(NEW_APP))

    # Strict check of key integration changes; no runtime/ML claim is made.
    assert '"best_unet_v3.pth"' in pipe_text
    assert '"live_v3"' in pipe_text
    assert "from sartrack_pipeline_v3 import run_pipeline" in app_text
    assert "Saved Esch demo: earlier U-Net v2 predictions." in app_text

    # Exclusive create means no original or existing v3 file can be overwritten.
    try:
        with NEW_PIPELINE.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(pipe_text)
        with NEW_APP.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(app_text)
    except Exception:
        # If app creation fails, remove only the new pipeline created by this run.
        if not NEW_APP.exists() and NEW_PIPELINE.exists():
            NEW_PIPELINE.unlink()
        raise

    print("CREATED:", NEW_PIPELINE.name)
    print("CREATED:", NEW_APP.name)
    print("V3 checkpoint found:", WEIGHTS.name, f"({WEIGHTS.stat().st_size / 1e6:.2f} MB)")
    print("V3 live outputs: outputs/live_v3/")
    print("Saved Esch demo remains v2 (unchanged).")
    print("Original app.py and sartrack_pipeline.py: UNCHANGED")
    print("V3 DASHBOARD FILES CREATED; syntax validated, runtime NOT YET TESTED")


if __name__ == "__main__":
    main()
