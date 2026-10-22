"""Redact machine-specific paths from QED_LEAF result files before publication.

Absolute paths of the project and of the user's home directory are replaced by
`<repo>` and `<home>`; digests, statuses and evidence are unchanged in content.
Note: the report's own content address (its file name / `report` id) was
computed by QED_LEAF over the unredacted report, so the redacted copy is a
publication copy, not the content-addressed original (which stays in the
git-ignored work directory).
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def _variants(p: Path) -> list[str]:
    s = str(p)
    out = {s, s.replace("\\", "/"), s.replace("\\", "\\\\"), json.dumps(s)[1:-1]}
    return sorted(out, key=len, reverse=True)


def redact(text: str) -> str:
    for v in _variants(PROJECT):
        text = re.sub(re.escape(v), "<repo>", text, flags=re.I)
    for v in _variants(Path.home()):
        text = re.sub(re.escape(v), "<home>", text, flags=re.I)
    return text


def sanitize_results() -> list[str]:
    changed = []
    for f in sorted((HERE / "results").glob("*")):
        if f.suffix == ".gz":
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                text = fh.read()
            new = redact(text)
            if new != text:
                with gzip.open(f, "wt", encoding="utf-8") as fh:
                    fh.write(new)
                changed.append(f.name)
        elif f.suffix == ".json":
            text = f.read_text(encoding="utf-8")
            new = redact(text)
            if new != text:
                f.write_text(new, encoding="utf-8")
                changed.append(f.name)
    return changed


if __name__ == "__main__":
    print("redacted:", sanitize_results())
