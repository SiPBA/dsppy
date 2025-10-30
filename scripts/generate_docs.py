#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def ensure_docs_skeleton(docs_dir: Path) -> None:
    (docs_dir / "_static").mkdir(parents=True, exist_ok=True)
    (docs_dir / "_templates").mkdir(parents=True, exist_ok=True)
    # Create minimal index if missing; conf.py is committed in repo
    index_path = docs_dir / "index.rst"
    if not index_path.exists():
        index_path.write_text(
            """
dsppy documentation
===================

Welcome to the dsppy documentation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
""".lstrip()
        )


def generate_api_docs(src_pkg_dir: Path, api_out_dir: Path) -> None:
    if api_out_dir.exists():
        shutil.rmtree(api_out_dir)
    api_out_dir.mkdir(parents=True, exist_ok=True)

    # Use sphinx-apidoc to generate reST stubs
    run(
        [
            sys.executable,
            "-m",
            "sphinx.ext.apidoc",
            "-o",
            str(api_out_dir),
            str(src_pkg_dir),
            "--force",
            "--no-toc",
        ]
    )

    # Create an API index toctree if missing
    index_rst = api_out_dir / "index.rst"
    if not index_rst.exists():
        # Collect modules
        entries = []
        for p in sorted(api_out_dir.glob("*.rst")):
            if p.name == "modules.rst" or p.name == "index.rst":
                continue
            entries.append(p.stem)
        index_rst.write_text(
            ("API Reference\n============\n\n" ".. toctree::\n   :maxdepth: 2\n\n")
            + "\n".join(f"   {name}" for name in entries)
            + "\n"
        )


def build_docs(docs_dir: Path) -> None:
    build_dir = docs_dir / "_build"
    if build_dir.exists():
        # Do not wipe all builders; html will be refreshed in place
        pass
    run(["sphinx-build", "-b", "html", str(docs_dir), str(build_dir / "html")])


def main() -> None:
    # scripts/ is directly under project root
    project_root = Path(__file__).resolve().parents[1]
    docs_dir = project_root / "docs"
    src_dir = project_root / "src"
    pkg_dir = src_dir / "dsppy"

    if not pkg_dir.exists():
        print(f"Package directory not found: {pkg_dir}", file=sys.stderr)
        sys.exit(1)

    ensure_docs_skeleton(docs_dir)
    api_dir = docs_dir / "api"
    generate_api_docs(pkg_dir, api_dir)
    build_docs(docs_dir)
    out = docs_dir / "_build" / "html"
    print(f"Docs built: {out}")


if __name__ == "__main__":
    main()


