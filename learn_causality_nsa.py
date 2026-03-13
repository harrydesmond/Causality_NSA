"""Run FCIT on NSA data and generate the PAG (Fig 4)."""

from __future__ import annotations

import os
import pickle
import re
import shutil
import subprocess
import time
from pathlib import Path

import graphviz as gviz
import numpy as np
import pandas as pd
import pytetrad.tools.TetradSearch as ts


PVAL_THRESHOLD = 0.01
TRUNC_LIMIT = 14
PENALTY_DISCOUNT = 50


def _resolve_input_pickle() -> Path:
    input_pickle = Path.cwd() / "nsa.pkl"
    if input_pickle.exists():
        return input_pickle.resolve()
    raise FileNotFoundError(f"Could not find {input_pickle}")


def _resolve_paper_plots_dir() -> Path:
    out_dir = Path("Plots_paper").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _tighten_dot(dot: str) -> str:
    lines = dot.splitlines()
    open_idx = None
    for idx, line in enumerate(lines):
        if "{" in line:
            open_idx = idx
            break
    if open_idx is not None:
        lines.insert(open_idx + 1, '  graph [margin=0, pad=0, label="", xlabel=""];')
    return "\n".join(lines)


def _relabel_dot(dot: str) -> str:
    table_start = '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">'
    table_end = "</TABLE>"
    mapping = {
        "ZDIST": table_start
        + '<TR><TD><B>Redshift</B></TD></TR><TR><TD><FONT POINT-SIZE="10">ZDIST</FONT></TD></TR>'
        + table_end,
        "log(B300)": table_start
        + '<TR><TD><B>Recent star formation</B></TD></TR><TR><TD><FONT POINT-SIZE="10">log(B300)</FONT></TD></TR>'
        + table_end,
        "log(MASS)": table_start
        + '<TR><TD><B>Stellar mass</B></TD></TR><TR><TD><FONT POINT-SIZE="10">log(MASS)</FONT></TD></TR>'
        + table_end,
        "ABSMAG": table_start
        + '<TR><TD><B>r-band luminosity</B></TD></TR><TR><TD><FONT POINT-SIZE="10">ABSMAG</FONT></TD></TR>'
        + table_end,
        "ELPETRO_BA": table_start
        + '<TR><TD><B>Axis ratio</B></TD></TR><TR><TD><FONT POINT-SIZE="10">ELPETRO_BA</FONT></TD></TR>'
        + table_end,
        "SERSIC_N": table_start
        + '<TR><TD><B>Morphology</B></TD></TR><TR><TD><FONT POINT-SIZE="10">SERSIC_N</FONT></TD></TR>'
        + table_end,
        "ELPETRO_TH50_R": table_start
        + '<TR><TD><B>Apparent size</B></TD></TR><TR><TD><FONT POINT-SIZE="10">ELPETRO_TH50_R</FONT></TD></TR>'
        + table_end,
    }

    lines = dot.splitlines()
    edge_re = re.compile(r'\s*".*"\s*->')
    insert_idx = None
    for idx, line in enumerate(lines):
        if edge_re.match(line):
            insert_idx = idx
            break
    if insert_idx is None:
        for idx, line in enumerate(lines):
            if line.strip() == "}":
                insert_idx = idx
                break
    if insert_idx is None:
        insert_idx = len(lines)

    node_lines = [f' "{name}" [label=<{html}>]; ' for name, html in mapping.items()]
    lines[insert_idx:insert_idx] = node_lines
    return "\n".join(lines)


def _crop_pdf(path: Path) -> None:
    pdfcrop = shutil.which("pdfcrop")
    if not pdfcrop or not path.exists():
        return
    tmp = path.with_name(path.stem + "_cropped.pdf")
    subprocess.run(
        [pdfcrop, "--margins", "0", str(path), str(tmp)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if tmp.exists():
        os.replace(tmp, path)


def main() -> None:
    input_pickle = _resolve_input_pickle()
    paper_plots_dir = _resolve_paper_plots_dir()

    print(f"Using p-value threshold: {PVAL_THRESHOLD}", flush=True)
    print(f"Using truncation limit: {TRUNC_LIMIT}", flush=True)
    print(f"Using penalty discount: {PENALTY_DISCOUNT}", flush=True)
    print(f"Using input: {input_pickle}", flush=True)

    with open(input_pickle, "rb") as f:
        data_dict = pickle.load(f)

    var_names = list(data_dict.keys())
    data = np.column_stack([data_dict[var] for var in var_names])
    data_df = pd.DataFrame(data, columns=var_names)

    start = time.time()
    search = ts.TetradSearch(data_df)
    search.set_verbose(True)
    search.use_basis_function_lrt(truncation_limit=TRUNC_LIMIT, alpha=PVAL_THRESHOLD)
    search.use_basis_function_bic(
        truncation_limit=TRUNC_LIMIT,
        penalty_discount=PENALTY_DISCOUNT,
    )
    search.run_fcit()
    print(f"Time taken to run FCIT: {round(time.time() - start, 3)}", flush=True)

    dot_text = _tighten_dot(search.get_dot())
    dot_text = _relabel_dot(dot_text)
    graph = gviz.Source(dot_text)

    out_base = paper_plots_dir / "data"
    pdf_path = Path(graph.render(filename=str(out_base), format="pdf", cleanup=True))
    _crop_pdf(pdf_path)
    print(f"Saved paper figure: {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
