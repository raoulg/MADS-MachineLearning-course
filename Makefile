.PHONY: help publish-fashion publish-flowers publish-imdb publish-all list-caches \
        build-eurosat build-dtd build-lesson1 publish-eurosat publish-dtd publish-lesson1 \
        build-heartbeats

# Publishing course caches to https://huggingface.co/pttrn-io
#
# The artefact folders carry a timestamp, so these targets pick the *newest* match rather
# than making you paste one. Add ARGS=--dry-run to any target to see the repo id and the
# generated dataset card without touching the hub.
#
#   make publish-fashion ARGS=--dry-run
#   make publish-all
#
# Lesson 1 needs eurosat (demo) and dtd (exercises), each encoded by several encoders so
# the ray search in 04_search.ipynb has an `encoder` axis to search over. Build once,
# publish once, and no student ever runs an encoder:
#
#   make build-lesson1        # ~15 min on an mps laptop, all encoders x both datasets
#   make publish-lesson1 ARGS=--dry-run
#   make publish-lesson1

ENCODER ?= mobilenet_v2_1.0_224
ARGS ?=

# Encoders for the lesson-1 caches: one strong, two weak. The spread is the point --
# 04_search.ipynb searches over this axis and finds it dominates every head knob.
# ENCODERS ?= facebook/dinov2-small microsoft/resnet-18 google/mobilenet_v2_1.0_224
ENCODERS ?= facebook/dinov2-base facebook/dinov2-large

# newest artefact folder matching a glob, or empty
latest = $(shell ls -dt artefacts/$(1) 2>/dev/null | head -1)

help:
	@echo "targets:"
	@echo "  list-caches      show the artefact folders these targets would pick"
	@echo "  publish-fashion  fashion-mnist x \$$ENCODER  (default: $(ENCODER))"
	@echo "  publish-flowers  oxford-flowers x \$$ENCODER"
	@echo "  publish-imdb     imdb x granite-embedding-small-english-r2"
	@echo "  publish-all      all three"
	@echo "  ENCODERS = $(ENCODERS)"
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


build-eurosat:
	@for enc in $(ENCODERS); do \
		echo "==> eurosat x $$enc"; \
		uv run python scripts/build_cache.py \
			--source tanganke/eurosat --encoder $$enc --tag eurosat \
			--n-train 6000 --n-test 2000 || exit 1; \
	done

build-dtd:
	@for enc in $(ENCODERS); do \
		echo "==> dtd x $$enc"; \
		uv run python scripts/build_cache.py \
			--source tanganke/dtd --encoder $$enc --tag dtd --full || exit 1; \
	done

# ECG heartbeats (PTB, MIT-BIH) from the private repo raoulg/mads-exam-24: no encoder, the
# 187 steps of a beat go into the cache as a (187, 1) sequence. Cloned on first use.
HEARTBEAT_SOURCE = artefacts/sources/mads-exam-24

$(HEARTBEAT_SOURCE):
	gh repo clone raoulg/mads-exam-24 $(HEARTBEAT_SOURCE)

build-heartbeats: | $(HEARTBEAT_SOURCE)
	uv run python scripts/build_heartbeat_cache.py

publish-eurosat:
	@for enc in $(ENCODERS); do \
		slug=$${enc##*/}; \
		echo "==> publishing eurosat x $$slug"; \
		uv run python scripts/publish_cache.py $(ARGS) \
			--dataset eurosat --source tanganke/eurosat \
			--train $$(ls -dt artefacts/*_eurosat_train_$${slug}_embed 2>/dev/null | head -1) \
			--test  $$(ls -dt artefacts/*_eurosat_test_$${slug}_embed 2>/dev/null | head -1) || exit 1; \
	done

publish-dtd:
	@for enc in $(ENCODERS); do \
		slug=$${enc##*/}; \
		echo "==> publishing dtd x $$slug"; \
		uv run python scripts/publish_cache.py $(ARGS) \
			--dataset dtd --source tanganke/dtd \
			--train $$(ls -dt artefacts/*_dtd_train_$${slug}_embed 2>/dev/null | head -1) \
			--test  $$(ls -dt artefacts/*_dtd_test_$${slug}_embed 2>/dev/null | head -1) || exit 1; \
	done
