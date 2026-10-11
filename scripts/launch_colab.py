"""Start the temporary password-protected Gradio workspace, not a model API."""

import os
from pathlib import Path
from spatialmemory.colab_ui import build_demo


def main():
    password = os.environ.get("SPATIALMEMORY_UI_PASSWORD")
    if not password or len(password) < 12:
        raise SystemExit("Set SPATIALMEMORY_UI_PASSWORD to at least 12 characters.")
    repo = Path(__file__).resolve().parents[1]
    root = Path(os.environ.get("SPATIALMEMORY_RUNS", "/content/spatialmemory-runs"))
    demo = build_demo(root, repo / "configs/phone_challenge.json")
    demo.launch(
        share=True,
        auth=("reviewer", password),
        auth_message="Sign in with the password you set in the Colab notebook.",
        max_file_size="512mb",
        show_error=False,
        enable_monitoring=False,
        strict_cors=True,
    )


if __name__ == "__main__":
    main()
