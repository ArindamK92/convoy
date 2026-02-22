SHELL := /bin/bash

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: help docs-install docs-serve docs-build docs-clean

help:
	@echo "Available targets:"
	@echo "  make docs-install  # Install documentation dependencies"
	@echo "  make docs-serve    # Serve docs locally at http://127.0.0.1:8000"
	@echo "  make docs-build    # Build static docs into site/"
	@echo "  make docs-clean    # Remove generated docs output (site/)"

docs-install:
	$(PIP) install -r docs/requirements.txt

docs-serve:
	$(PYTHON) -m mkdocs serve

docs-build:
	$(PYTHON) -m mkdocs build

docs-clean:
	rm -rf site
