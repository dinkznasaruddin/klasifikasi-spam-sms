#!/usr/bin/env bash
set -euo pipefail
# Pastikan pandoc terpasang: https://pandoc.org/installing.html
# Unduh file CSL APA (jika belum ada)
mkdir -p paper
if [ ! -f paper/apa.csl ]; then
  curl -L -o paper/apa.csl https://www.zotero.org/styles/apa
fi

# Siapkan reference.docx opsional untuk kontrol gaya (font, spasi, margin)
# Jika belum ada, buat dari template default Pandoc agar bisa diedit manual di Word/LibreOffice.
if [ ! -f paper/reference.docx ]; then
  pandoc --print-default-data-file=reference.docx > paper/reference.docx || true
fi

# Ekspor ke DOCX dengan sitasi APA
# Gunakan reference.docx bila tersedia untuk mengontrol gaya dokumen
REF_OPTS=()
if [ -f paper/reference.docx ]; then
  REF_OPTS=(--reference-doc=paper/reference.docx)
fi

pandoc paper/paper.md \
  --from markdown+yaml_metadata_block \
  --citeproc \
  --csl paper/apa.csl \
  --bibliography paper/refs.bib \
  --resource-path=. \
  "${REF_OPTS[@]}" \
  -o paper/paper_APA.docx

echo "Selesai: paper/paper_APA.docx"