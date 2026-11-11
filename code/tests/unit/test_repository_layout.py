from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]


def documents():
    yield ROOT / "README.md"
    for directory in ("scripts", "tests", "prompts"):
        yield from (ROOT / directory).rglob("*.md")


def test_local_document_links_resolve():
    missing = []
    for document in documents():
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            target = target.strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (document.parent / unquote(target)).exists():
                missing.append(f"{document}: {target}")
    assert not missing, "Broken local links:\n" + "\n".join(missing)


def heading_ids(document):
    counts = {}
    result = set()
    fenced = False
    for line in document.read_text(encoding="utf-8").splitlines():
        if line.startswith(("```", "~~~")):
            fenced = not fenced
            continue
        match = re.match(r"^#{1,6} (.+)", line)
        if fenced or not match:
            continue
        slug = "".join(
            c for c in match[1].strip().lower()
            if c in "-_ " or unicodedata.category(c)[0] in "LN"
        ).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(slug + (f"-{count}" if count else ""))
    return result


def test_document_heading_links_resolve():
    missing = []
    for document in documents():
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if "://" in target or "#" not in target:
                continue
            path, fragment = unquote(target.strip("<>")).split("#", 1)
            destination = document.parent / path if path else document
            if destination.is_file() and fragment not in heading_ids(destination):
                missing.append(f"{document}: {target}")
    assert not missing, "Broken heading links:\n" + "\n".join(missing)


def test_documented_python_commands_exist():
    references = []
    pattern = re.compile(r"(?<![\w/.-])(?:app/)?((?:scripts|tests)/[\w/.-]+\.py)")
    for document in documents():
        references.extend(pattern.findall(document.read_text(encoding="utf-8")))
    assert references
    missing = sorted({name for name in references if not (ROOT / name).is_file()})
    assert not missing, f"Outdated command/test paths: {missing}"


def test_script_groups_are_explicit():
    scripts = ROOT / "scripts"
    assert {p.name for p in scripts.glob("*.py")} == {"_cli.py", "fetch_models.py"}
    groups = {p.parent.name for p in scripts.glob("*/*.py")}
    assert groups == {"asr", "corpus", "matcher", "quality"}


def test_requirements_delegate_to_package_metadata():
    for filename, expected in (("requirements.txt", "-e ."), ("requirements-dev.txt", "-e .[dev]")):
        lines = [
            line.strip()
            for line in (ROOT / filename).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert lines == [expected], "Keep dependency declarations in pyproject.toml"


def test_matcher_package_keeps_required_projection_assets():
    from willspeak.matcher import engine, scoring, training
    from willspeak.paths import PROJECTION_DIR

    assert callable(engine.Matcher) and callable(scoring.rank_candidates) and callable(training.lda_fit)
    assert (PROJECTION_DIR / "projection.npz").is_file()
    assert not (PROJECTION_DIR / "projection_tts.npz").exists()
