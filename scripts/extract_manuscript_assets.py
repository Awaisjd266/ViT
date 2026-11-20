"""Export user-authorized manuscript figures/tables with explicit claim provenance."""
import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


def extract(source: Path, output: Path):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(source) as z:
        doc = ET.fromstring(z.read("word/document.xml"))
        tables = []
        for table in doc.findall(".//w:tbl", ns):
            rows = []
            for row in table.findall("w:tr", ns):
                rows.append(["".join(t.text or "" for t in c.findall(".//w:t", ns))
                             for c in row.findall("w:tc", ns)])
            if rows and rows[0][0].strip() in {"Dataset", "Model", "Metric"}:
                tables.append(rows)
        for i, rows in enumerate(tables, 1):
            with (output / f"manuscript_table_{i}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["origin", *[c.strip() for c in rows[0]]])
                writer.writerows(["unverified_manuscript_claim", *[c.strip() for c in row]] for row in rows[1:])
        figures = []
        for name in z.namelist():
            if name.startswith("word/media/"):
                dest = output / Path(name).name
                dest.write_bytes(z.read(name))
                figures.append(dest.name)
        manifest = {"source_title": "Enhancing Road Pavement Crack Detection Using a Vision Transformer Approach",
                    "authors": ["Muhammad Awais Javeed", "Zijie Zhou", "Dawei Li"],
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "origin": "user-supplied manuscript; not independently reproduced",
                    "figures": figures, "tables": len(tables)}
        (output / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, default=Path("docs/manuscript"))
    a = p.parse_args()
    extract(a.source, a.output)
