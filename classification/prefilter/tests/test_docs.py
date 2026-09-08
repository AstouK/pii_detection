"""
Consistency checks across the repository's three README files.

Documentation drifts silently: a command gets renamed, a report moves, a metric
is quoted from memory, and nothing fails until someone follows the instructions
and they do not work. These checks turn the parts of that which are mechanically
verifiable into test failures.

They deliberately do not check prose quality. They check the things that are
either true or false:

    * every documented `python -m` target imports
    * every relative link resolves
    * every fenced-block path that looks like a repo path exists
    * shared constants agree across files
    * the caveat that stops the headline result being misquoted is present
    * no licence text was added (the repository owner rules it out)

Run:

    python -m pytest classification/prefilter/tests/test_docs.py -q
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

README_FILES = {
    "root": REPO_ROOT / "README.md",
    "classification": REPO_ROOT / "classification" / "README.md",
    "prefilter": REPO_ROOT / "classification" / "prefilter" / "README.md",
}


def _read(name: str) -> str:
    path = README_FILES[name]

    if not path.exists():
        pytest.skip(f"{path} does not exist")

    return path.read_text(encoding="utf-8")


ALL_READMES = list(README_FILES)


# ─────────────────────────────────────────────────────────────
# Licence
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_READMES)
def test_no_licence_section_was_added(name: str):
    """
    The repository owner explicitly ruled out adding licence text.

    A heading or a shields.io licence badge — not the word appearing inside a
    sentence, which is legitimate (MEDICAL_LICENSE is one of the twelve entity
    labels, and it appears in every column table).
    """

    text = _read(name)

    heading = re.search(r"^#{1,6}\s*licen[cs]e\b", text, re.IGNORECASE | re.MULTILINE)
    assert heading is None, f"{name} README has a licence heading"

    badge = re.search(r"shields\.io/[^)\s]*licen[cs]e", text, re.IGNORECASE)
    assert badge is None, f"{name} README has a licence badge"


def test_no_license_file_was_created():
    for candidate in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        assert not (REPO_ROOT / candidate).exists(), f"{candidate} was created"


# ─────────────────────────────────────────────────────────────
# Documented commands
# ─────────────────────────────────────────────────────────────

MODULE_PATTERN = re.compile(r"python -m ([a-zA-Z_][\w.]*)")

#: Not importable as modules, and not meant to be.
MODULE_ALLOWLIST = {"pytest", "venv", "pip"}


@pytest.mark.parametrize("name", ALL_READMES)
def test_documented_python_modules_exist(name: str):
    """
    Every `python -m <module>` in the docs must resolve to a real module.
    """

    text = _read(name)

    missing = []

    for module in sorted(set(MODULE_PATTERN.findall(text))):
        if module.split(".")[0] in MODULE_ALLOWLIST:
            continue

        try:
            if importlib.util.find_spec(module) is None:
                missing.append(module)
        except (ImportError, ModuleNotFoundError, ValueError):
            missing.append(module)

    assert not missing, f"{name} README documents non-existent modules: {missing}"


CONSOLE_SCRIPTS = {"classify", "evaluate", "update-dataset"}


def test_console_scripts_match_pyproject():
    """
    The scripts the docs promise are the scripts pyproject actually installs.
    """

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    declared = set(
        re.findall(r"^([\w-]+)\s*=\s*\"[\w.]+:\w+\"", pyproject, re.MULTILINE)
    )

    assert CONSOLE_SCRIPTS <= declared, (
        f"pyproject declares {declared}, docs assume {CONSOLE_SCRIPTS}"
    )


# ─────────────────────────────────────────────────────────────
# Links and paths
# ─────────────────────────────────────────────────────────────

LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


@pytest.mark.parametrize("name", ALL_READMES)
def test_relative_links_resolve(name: str):
    """
    Relative links between the READMEs must point at files that exist.
    """

    text = _read(name)
    base = README_FILES[name].parent

    broken = []

    for target in LINK_PATTERN.findall(text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue

        path = (base / target.split("#")[0]).resolve()

        if not path.exists():
            broken.append(target)

    assert not broken, f"{name} README has broken relative links: {broken}"


#: Repo paths named in prose or code blocks. Anything matching this shape is
#: expected to exist; a directory may be git-ignored but must still be a path
#: the project actually uses.
REPO_PATH_PATTERN = re.compile(
    r"`(classification/[\w./<>-]+|config/[\w./-]+|models/[\w./-]+)`"
)

#: Paths that legitimately do not exist in this working tree:
#:   <run_id> etc.     placeholders
#:   results/          run output, git-ignored
#:   artifacts/        checkpoints, git-ignored
#:   external/         datasets fetched from another work package's branch
#:   data_generation/  lives on Sonja's branch; the docs cite it as the source
#:                     of the 1,400-row dataset, and it is correct that it is
#:                     absent here — vendoring it would create merge conflicts
PATH_ALLOWLIST_PATTERN = re.compile(
    r"<[\w-]+>|\*|results/|artifacts/|external/|data_generation/"
)


@pytest.mark.parametrize("name", ALL_READMES)
def test_referenced_repo_paths_exist(name: str):
    """
    A backticked repo path in the docs should be a real path.
    """

    text = _read(name)

    missing = []

    for target in sorted(set(REPO_PATH_PATTERN.findall(text))):
        if PATH_ALLOWLIST_PATTERN.search(target):
            continue

        if not (REPO_ROOT / target).exists():
            missing.append(target)

    assert not missing, f"{name} README references missing paths: {missing}"


# ─────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────

def test_baseline_recall_agrees_everywhere():
    """
    The rule-based baseline is the number the whole work package is judged
    against. Every file that quotes it must quote the same value, and it must
    match the constant the training code actually uses.
    """

    from classification.prefilter.train import BASELINE_RECALL

    assert BASELINE_RECALL == 0.9833

    for name in ALL_READMES:
        text = _read(name)

        quoted = set(re.findall(r"0\.98\d*", text))
        wrong = {
            value
            for value in quoted
            if value not in {"0.9833", "0.98"}
        }

        assert not wrong, f"{name} README quotes a different baseline: {wrong}"


def test_entity_labels_are_quoted_correctly():
    """
    The twelve labels are an interface, not prose: the evaluation derives its
    entity vocabulary from these exact spellings. Any README that lists them
    must list all twelve.
    """

    from classification.prefilter.config import ENTITY_LABELS

    text = _read("prefilter")

    missing = [label for label in ENTITY_LABELS if label not in text]

    assert not missing, f"prefilter README omits entity labels: {missing}"


# ─────────────────────────────────────────────────────────────
# The caveat
# ─────────────────────────────────────────────────────────────

CAVEAT_MARKERS = [
    "degenerate",
    "separable",
]


def test_headline_result_keeps_its_caveat():
    """
    The pilot's 0.0% routing rate is a degenerate outcome, not a capability.

    Any README quoting it has to carry the reason in the same document, or the
    number gets repeated in a meeting as though it meant something.
    """

    for name in ALL_READMES:
        text = _read(name).lower()

        if "0.0%" not in text and "0%" not in text:
            continue

        assert any(marker in text for marker in CAVEAT_MARKERS), (
            f"{name} README quotes the 0% routing rate without the caveat "
            f"(expected one of {CAVEAT_MARKERS})"
        )


# ─────────────────────────────────────────────────────────────
# Markdown structure
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_READMES)
def test_code_fences_are_balanced(name: str):
    """
    An unclosed fence swallows the rest of the page when GitHub renders it.

    This is not hypothetical: classification/README.md shipped with a truncated
    `pip install -r` and an unterminated fence.
    """

    text = _read(name)

    fences = re.findall(r"^```", text, re.MULTILINE)

    assert len(fences) % 2 == 0, (
        f"{name} README has an unbalanced code fence ({len(fences)} markers)"
    )


@pytest.mark.parametrize("name", ALL_READMES)
def test_mermaid_blocks_are_declared(name: str):
    """
    A mermaid block only renders when the fence carries the language tag.
    """

    text = _read(name)

    if "flowchart" not in text and "graph TD" not in text:
        pytest.skip(f"{name} README has no mermaid diagram")

    assert "```mermaid" in text, (
        f"{name} README has mermaid syntax in a fence without the language tag"
    )


@pytest.mark.parametrize("name", ALL_READMES)
def test_details_blocks_are_closed(name: str):
    """
    An unclosed <details> hides everything after it.
    """

    text = _read(name)

    assert text.count("<details") == text.count("</details>"), (
        f"{name} README has an unclosed <details> block"
    )


# ─────────────────────────────────────────────────────────────
# Self-consistency
# ─────────────────────────────────────────────────────────────

TEST_COUNT_PATTERN = re.compile(r"(\d+)\s+(?:checks|tests)\b", re.IGNORECASE)


def _collected_test_count() -> int | None:
    """
    How many test cases the suite actually collects, parametrisation included.

    Asked of pytest rather than counted from `def test_` lines: parametrised
    cases expand, so a source count is a floor rather than the number a reader
    sees. Returns None when collection cannot be run, so the check skips instead
    of failing for an unrelated reason.
    """

    import subprocess
    import sys

    tests_dir = Path(__file__).resolve().parent

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(tests_dir), "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=REPO_ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    match = re.search(r"(\d+) tests? collected", result.stdout)

    return int(match.group(1)) if match else None


@pytest.mark.parametrize("name", ALL_READMES)
def test_documented_test_count_is_current(name: str):
    """
    A README that advertises "N checks" must advertise the real N.

    This drifted the moment this file was added: the root README promised 21
    routing-logic checks for a directory that had just grown a second suite.
    Deriving the number from a real collection means the claim cannot go stale
    again without something failing.
    """

    text = _read(name)

    claimed = {int(value) for value in TEST_COUNT_PATTERN.findall(text)}

    if not claimed:
        pytest.skip(f"{name} README makes no test-count claim")

    collected = _collected_test_count()

    if collected is None:
        pytest.skip("could not collect the suite to verify the count")

    # A README may cite the whole suite or one file within it, so any claim from
    # 1 up to the collected total is legitimate; a claim above it is stale.
    stale = {value for value in claimed if value > collected}

    assert not stale, (
        f"{name} README claims {stale} checks, but the suite collects "
        f"{collected}. Update the README or the claim is a lie to the next "
        f"person who runs it."
    )


MERMAID_BLOCK_PATTERN = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)

#: `&lt;` and `&gt;` inside a mermaid edge label render unreliably under
#: GitHub's htmlLabels renderer — sometimes as the character, sometimes as the
#: literal entity, sometimes swallowing the rest of the label. Spell the
#: comparison out, or use ≤ / ≥, which are plain characters.
HTML_ENTITY_PATTERN = re.compile(r"&(lt|gt|amp|quot|#\d+);")


@pytest.mark.parametrize("name", ALL_READMES)
def test_mermaid_labels_avoid_html_entities(name: str):
    """
    Mermaid diagrams must not depend on HTML entities to render.
    """

    text = _read(name)

    offenders = []

    for block in MERMAID_BLOCK_PATTERN.findall(text):
        for line in block.splitlines():
            if HTML_ENTITY_PATTERN.search(line):
                offenders.append(line.strip())

    assert not offenders, (
        f"{name} README uses HTML entities inside a mermaid diagram, which "
        f"renders unreliably on GitHub: {offenders}"
    )
