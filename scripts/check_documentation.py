"""Validate relative Markdown paths, config schemas and example CLI availability."""
import re
from pathlib import Path

from crackvit.cli import parser
from crackvit.config import load_config
from crackvit.utils import write_json


def main():
    failures, checked = [], 0
    for path in [Path("README.md"), *Path("docs").glob("*.md"), *Path("datasets").glob("*.md")]:
        text = path.read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if link.startswith(("https:", "http:", "#")):
                continue
            target = link.split("#")[0]
            checked += 1
            if not (path.parent / target).exists():
                failures.append({"source": str(path), "missing": target})
    configs = list(Path("configs").glob("*.yaml"))
    for p in configs:
        load_config(p)
    cli = parser()
    # These are actual subcommands shown by the single integrated parser.
    required = {"train", "evaluate", "infer", "audit", "prepare", "doctor", "suite", "tune"}
    options = next(a for a in cli._actions if getattr(a, "choices", None)).choices
    if not required <= set(options):
        failures.append({"missing_cli_commands": sorted(required - set(options))})
    result = {"relative_links_checked": checked, "configurations_validated": len(configs),
              "cli_commands": sorted(options), "failures": failures, "passed": not failures}
    write_json(Path("reports/documentation_check.json"), result)
    if failures:
        raise SystemExit(1)
    print(result)


if __name__ == "__main__":
    main()
