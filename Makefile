PYTHON	?= python3.13
PIP		?= pip3.13
CONFIG	?= citrus_exp6

# The finalized datasets (keep in sync with src/configs/*.py).
CONFIGS	= citrus_exp6 tomato_nir grape_berry apple_samnegard mango_composition biofood_safou_region

venv:
	$(PYTHON) -m venv venv
	venv/bin/$(PYTHON) -m pip install --upgrade pip setuptools

.PHONY: setup
setup: venv
	venv/bin/$(PIP) install -r requirements.txt

# ---- single dataset (run each step separately) ----

# Comparison + evaluation (+ TSTR downstream if the config enables it).
.PHONY: run
run:
	venv/bin/$(PYTHON) -m src.main $(CONFIG)

# Same-model near/far transfer (requires LORO_GROUP in the config).
.PHONY: loro
loro:
	venv/bin/$(PYTHON) -m src.run_loro $(CONFIG)

# Train/test generalization check for one dataset.
.PHONY: check-overfit
check-overfit:
	venv/bin/$(PYTHON) -m src.check_generalization $(CONFIG)

# ---- all datasets at once ----

# Comparison on every dataset in turn.
.PHONY: run-all
run-all:
	@for cfg in $(CONFIGS); do \
		echo "==================== run: $$cfg ===================="; \
		venv/bin/$(PYTHON) -m src.main $$cfg || exit 1; \
	done

# Near/far transfer on every dataset that defines a group (others just log that
# there is nothing to hold out).
.PHONY: loro-all
loro-all:
	@for cfg in $(CONFIGS); do \
		echo "==================== loro: $$cfg ===================="; \
		venv/bin/$(PYTHON) -m src.run_loro $$cfg || exit 1; \
	done

# Everything, in order: comparisons -> near/far.
.PHONY: all
all: run-all loro-all
	@echo "==================== done: reports/ + results/_summary/ ===================="
