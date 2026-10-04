"""Regression guard: the CLI config must not hardcode model IDs."""

import re
from pathlib import Path

import venice_ai.cli.config as cfg


def test_default_config_has_no_model_keys():
    defaults = cfg.DEFAULT_CONFIG["defaults"]
    forbidden = {
        "chat_model",
        "image_model",
        "tts_model",
        "stt_model",
        "embedding_model",
        "decision_model",
        "video_t2v_model",
        "video_i2v_model",
    }
    assert forbidden.isdisjoint(defaults), (
        f"DEFAULT_CONFIG must not hardcode model IDs: {forbidden & set(defaults)}"
    )


def test_no_hidream_literal_in_cli_source():
    cli_dir = Path(cfg.__file__).parent
    hits = [str(p) for p in cli_dir.rglob("*.py") if re.search(r'["\']hidream["\']', p.read_text())]
    assert not hits, f"Retired 'hidream' literal found in: {hits}"
