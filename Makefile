# Makefile for the Multi-verse project

# --- Docker Image Builds ---

.PHONY: build-all
build-all: build-pca build-multivi build-mowgli
	@echo "All model images built."

.PHONY: build-pca
build-pca:
	@echo "Building PCA image..."
	docker build -f docker/pca.Dockerfile -t multiverse-pca .

.PHONY: build-multivi
build-multivi:
	@echo "Building MultiVI image..."
	docker build -f docker/multivi.Dockerfile -t multiverse-multivi .

.PHONY: build-mowgli
build-mowgli:
	@echo "Building Mowgli image..."
	docker build -f docker/mowgli.Dockerfile -t multiverse-mowgli .


# --- Orchestrator Runner ---

# Define some default paths for the runner
# Users should override these as needed
INPUT_DIR ?= ./sample_data/input
OUTPUT_DIR ?= ./results
MODELS_TO_RUN ?= pca multivi mowgli

.PHONY: run-all
run-all:
	@echo "Running orchestrator for models: $(MODELS_TO_RUN)"
	python -m multiverse.runner.cli --models $(MODELS_TO_RUN) --input $(INPUT_DIR) --output $(OUTPUT_DIR)

.PHONY: clean
clean:
	@echo "Cleaning up output directory..."
	rm -rf $(OUTPUT_DIR)
