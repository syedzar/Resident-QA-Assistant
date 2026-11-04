"""Stage 1 smoke test: the repo has the layout the pipeline and CI expect."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_expected_folders_exist():
    for folder in ["databricks/notebooks", "src/rcqa", "app", "eval", "tests", "data/sources"]:
        assert (ROOT / folder).is_dir(), f"missing folder: {folder}"


def test_env_file_is_gitignored():
    assert ".env" in (ROOT / ".gitignore").read_text().splitlines()


def test_raw_data_is_gitignored():
    assert "data/raw/*" in (ROOT / ".gitignore").read_text().splitlines()
