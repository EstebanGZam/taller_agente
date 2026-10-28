"""Launch the SECOP eligibility interface."""

from __future__ import annotations

import os

from secop_agent import settings
from secop_agent.web_app import build_demo


def main(argv: list[str] | None = None) -> int:
    settings.load_environment()
    build_demo().queue().launch(share="COLAB_RELEASE_TAG" in os.environ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
