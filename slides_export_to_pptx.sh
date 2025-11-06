#!/usr/bin/env bash
set -euo pipefail
mkdir -p slides

# Siapkan template PowerPoint (reference.pptx) jika belum ada
if [ ! -f slides/reference.pptx ]; then
  pandoc --print-default-data-file=reference.pptx > slides/reference.pptx || true
fi

# Ekspor Markdown ke PPTX
pandoc slides/slides_paper.md \
  --from markdown+yaml_metadata_block \
  --resource-path=.:artifacts:slides \
  --slide-level=1 \
  --reference-doc=slides/reference.pptx \
  -t pptx \
  -o slides/paper_presentation.pptx

echo "Selesai: slides/paper_presentation.pptx"
