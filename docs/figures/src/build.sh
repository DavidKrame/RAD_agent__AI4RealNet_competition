#!/bin/sh
# Rebuild every diagram in docs/figures from its TikZ source, in a light and a dark variant.
# Requires pdflatex and pdftocairo (poppler).
set -e
cd "$(dirname "$0")"
for f in *.tex; do
  [ "$f" = style.tex ] && continue
  n=${f%.tex}
  pdflatex -interaction=nonstopmode -halt-on-error -jobname="${n}_light" "\input{$f}" >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error -jobname="${n}_dark" "\def\darktheme{}\input{$f}" >/dev/null
  for t in light dark; do
    pdftocairo -png -r 220 -transp -singlefile "${n}_${t}.pdf" "../${n}_${t}"
  done
done
rm -f ./*.aux ./*.log ./*.pdf
