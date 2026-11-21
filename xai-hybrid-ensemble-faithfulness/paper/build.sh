#!/usr/bin/env bash
# Regenerates figures, every number/table and the supplement from ../results, then compiles (pdflatex + IEEEtran).
set -e; cd "$(dirname "$0")"
python make_figs.py && python make_ieee.py && python make_supp.py && node mkrefs.js
python - <<'PY'
import re
def inline(f):
    return re.sub(r"\\input\{([^}]+)\}", lambda m: inline(m.group(1)), open(f).read())
open("paper.tex","w").write(inline("main.tex"))
PY
for i in 1 2 3; do pdflatex -interaction=nonstopmode paper.tex >/dev/null || true; done
for i in 1 2; do pdflatex -interaction=nonstopmode supplement.tex >/dev/null || true; done
echo "built paper.pdf and supplement.pdf"
