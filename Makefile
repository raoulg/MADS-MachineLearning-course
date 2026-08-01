.PHONY: help publish-fashion publish-flowers publish-imdb publish-all list-caches

# Publishing course caches to https://huggingface.co/pttrn-io
#
# The artefact folders carry a timestamp, so these targets pick the *newest* match rather
# than making you paste one. Add ARGS=--dry-run to any target to see the repo id and the
# generated dataset card without touching the hub.
#
#   make publish-fashion ARGS=--dry-run
#   make publish-all

ENCODER ?= mobilenet_v2_1.0_224
ARGS ?=

# newest artefact folder matching a glob, or empty
latest = $(shell ls -dt artefacts/$(1) 2>/dev/null | head -1)

help:
	@echo "targets:"
	@echo "  list-caches      show the artefact folders these targets would pick"
	@echo "  publish-fashion  fashion-mnist x \$$ENCODER  (default: $(ENCODER))"
	@echo "  publish-flowers  oxford-flowers x \$$ENCODER"
	@echo "  publish-imdb     imdb x granite-embedding-small-english-r2"
	@echo "  publish-all      all three"
	@echo ""
	@echo "add ARGS=--dry-run to print the card without uploading"

list-caches:
	@echo "ENCODER = $(ENCODER)"
	@echo "fashion train : $(call latest,*_fashion_train_$(ENCODER)_embed)"
	@echo "fashion test  : $(call latest,*_fashion_test_$(ENCODER)_embed)"
	@echo "flowers train : $(call latest,*_flowers_train_$(ENCODER)_embed)"
	@echo "flowers test  : $(call latest,*_flowers_test_$(ENCODER)_embed)"
	@echo "imdb train    : $(call latest,*_imdb_train_embed)"
	@echo "imdb test     : $(call latest,*_imdb_test_embed)"

publish-fashion:
	uv run python scripts/publish_cache.py $(ARGS) \
		--dataset fashion-mnist \
		--source zalando-datasets/fashion_mnist \
		--train $(call latest,*_fashion_train_$(ENCODER)_embed) \
		--test  $(call latest,*_fashion_test_$(ENCODER)_embed)

publish-flowers:
	uv run python scripts/publish_cache.py $(ARGS) \
		--dataset oxford-flowers \
		--source nkirschi/oxford-flowers \
		--train $(call latest,*_flowers_train_$(ENCODER)_embed) \
		--test  $(call latest,*_flowers_test_$(ENCODER)_embed)

# IMDB folders have no encoder slug yet — 07_vectors predates that convention, and
# raoulg/vectormesh#1 will make the library add it for everyone. Update this glob then.
publish-imdb:
	uv run python scripts/publish_cache.py $(ARGS) \
		--dataset imdb \
		--source stanfordnlp/imdb \
		--column embed \
		--train $(call latest,*_imdb_train_embed) \
		--test  $(call latest,*_imdb_test_embed)

publish-all: publish-fashion publish-flowers publish-imdb
