PY ?= python3
PROVERB_PDF ?= references/proverbe-kikongo.pdf
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: venv data bago laman proverbs audit smoke sweep table clean-runs

venv:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q -U pip
	$(BIN)/pip install -q -r requirements.txt
	$(BIN)/pip install -q -e .

## Build every dataset. CPU + network only -- do this on the laptop, not the GPU.
data:
	$(BIN)/python -m pivotkk prepare

## Extract reference lexicons from the Bago dictionary (needs the PDF + poppler).
bago:
	$(BIN)/python scripts/extract_bago.py "$(BAGO_PDF)" --out data/lexicons

## OCR Laman (1936) into a second, independent Kikongo lexicon.
## 1,276 scanned pages; trial a range first: make laman LAMAN_PAGES=100-140
laman:
	$(BIN)/python scripts/ocr_laman.py "$(LAMAN_PDF)" \
	  $(if $(LAMAN_PAGES),--pages $(LAMAN_PAGES),) \
	  --out data/lexicons/laman_kikongo.txt

## Extract the Kunzika proverb dictionary into a kg/pt/fr/en parallel dataset.
## 362 scanned pages, OCR'd twice (~25 min the first time, then cached).
## Trial a range first: make proverbs PROVERB_PAGES=31-60
proverbs:
	$(BIN)/python scripts/extract_proverbs.py "$(PROVERB_PDF)" \
	  $(if $(PROVERB_PAGES),--pages $(PROVERB_PAGES),) \
	  --out data/proverbs

## Which language is each corpus? Run before spending money on training.
audit:
	$(BIN)/python -m pivotkk audit

## Verify the full train+eval loop on tiny data. ~2 min. Always run before a sweep.
smoke:
	$(BIN)/python -m pivotkk train configs/smol/lin_smol_seed13.yaml --smoke
	$(BIN)/python -m pivotkk eval runs/lin_smol_seed13_smoke --limit 20

## The core sweep: control + 3 pivots x 3 sizes.
sweep:
	bash scripts/run_sweep.sh

table:
	$(BIN)/python -m pivotkk table

clean-runs:
	rm -rf runs/*
