.PHONY: help study page share dashboard rehearsal optuna-dashboard check-lesson check-hf-auth publish-fashion publish-flowers publish-imdb publish-all list-caches \
        build-eurosat build-dtd build-lesson1 publish-eurosat publish-dtd publish-lesson1 \
        build-heartbeats publish-heartbeats

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

# The class's runs (docs/submit-runs.md, docs/run-protocol.md). Make takes variables, not
# flags, so the lesson goes in as LESSON=<N>; ARGS passes anything else to the script.
#
#   make share LESSON=1                    # submit your runs as a pull request
#   make share LESSON=1 ARGS=--list        # show what would be submitted, change nothing
#   make dashboard LESSON=1                # the class's runs: the class page, and optuna-dashboard on them
#   make dashboard LESSON=2 METRIC=recall/F  # the class page read on another logged metric

LESSON ?=
METRIC ?=

check-lesson:
	@test -n "$(LESSON)" || { echo "name the lesson: make $(MAKECMDGOALS) LESSON=<N>"; exit 2; }

check-hf-auth:
	@case " $(ARGS) " in \
		*" --dry-run "*) ;; \
		*) uv run hf auth whoami >/dev/null 2>&1 || { echo "Hugging Face login required; starting hf auth login."; uv run hf auth login; } ;; \
	esac

share: check-lesson
	uv run scripts/submit_runs.py --lesson $(LESSON) $(ARGS)

dashboard: check-lesson
	uv run scripts/class_analysis.py --lesson $(LESSON) --open $(if $(METRIC),--metric $(METRIC))
	uv run scripts/collect_runs.py --lesson $(LESSON) --no-pull --ui $(ARGS)

# A rehearsal of the class on this machine: four students in clones of their own, six
# studies that build on each other, handed in to a local class repository, and the
# teacher's class page of what was merged (scripts/class_rehearsal.py). Nothing reaches
# GitHub. ARGS="--minutes 2" gives each study two minutes.
#
#   make rehearsal

rehearsal:
	uv run scripts/class_rehearsal.py $(ARGS)

# The Optuna studies of notebooks/1_representation_and_head/05_exercises.ipynb, in
# optuna-dashboard on http://127.0.0.1:$(OPTUNA_PORT). The store exists once that notebook
# has run.
#
#   make optuna-dashboard
#   make optuna-dashboard OPTUNA_PORT=8089

OPTUNA_DIR ?= artefacts/optuna
OPTUNA_PORT ?= 8088

optuna-dashboard:
	@test -f $(OPTUNA_DIR)/studies.db || { echo "no studies yet: run notebooks/1_representation_and_head/05_exercises.ipynb first"; exit 2; }
	uv run optuna-dashboard sqlite:///$(OPTUNA_DIR)/studies.db \
		--artifact-dir $(OPTUNA_DIR)/artifacts --port $(OPTUNA_PORT) $(ARGS)

# Your study over a design of your own: presentations/your-study.html, and
# docs/your-study.md for an assistant. Part 1 of the lesson's exercise notebook runs
# first; it measures what one trial costs here.
#
#   make study            # lesson 1
#   make study LESSON=2

STUDY_SCRIPT = $(wildcard notebooks/$(or $(LESSON),1)_*/exercises/study.py)

study:
	@test -n "$(STUDY_SCRIPT)" || { echo "lesson $(or $(LESSON),1) has no exercises/study.py"; exit 2; }
	uv run $(STUDY_SCRIPT)

# Your page of a lesson, from your own store: artefacts/study-page/lesson<N>/index.html.
# What is yours on it comes from notebooks/<N>_*/exercises/page.toml.
#
#   make page LESSON=1
#   make page LESSON=1 ARGS="--submitted <login>"   # from a login's submitted runs

page: check-lesson
	uv run scripts/study_page.py --lesson $(LESSON) --open $(ARGS)

# newest artefact folder matching a glob, or empty
latest = $(shell ls -dt artefacts/$(1) 2>/dev/null | head -1)

help:
	@echo "targets:"
	@echo "  study [LESSON=<N>]    run your study over your design (notebooks/<N>_*/exercises; lesson 1 by default)"
	@echo "  page LESSON=<N>       open your page of lesson N: your ideas, design, search and results"
	@echo "  share LESSON=<N>      submit your runs of lesson N (ARGS=--list to preview)"
	@echo "  dashboard LESSON=<N>  open the class page of lesson N, and optuna-dashboard on the class's runs (METRIC=<name> to read another metric)"
	@echo "  rehearsal             rehearse the class here: four students, six studies, the class page (ARGS=\"--minutes <N>\")"
	@echo "  optuna-dashboard      open optuna-dashboard on your Optuna studies (05_exercises)"
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

publish-fashion: check-hf-auth
	uv run python scripts/publish_cache.py $(ARGS) \
		--dataset fashion-mnist \
		--source zalando-datasets/fashion_mnist \
		--train $(call latest,*_fashion_train_$(ENCODER)_embed) \
		--test  $(call latest,*_fashion_test_$(ENCODER)_embed)

publish-flowers: check-hf-auth
	uv run python scripts/publish_cache.py $(ARGS) \
		--dataset oxford-flowers \
		--source nkirschi/oxford-flowers \
		--train $(call latest,*_flowers_train_$(ENCODER)_embed) \
		--test  $(call latest,*_flowers_test_$(ENCODER)_embed)

# IMDB folders have no encoder slug yet — 07_vectors predates that convention, and
# raoulg/vectormesh#1 will make the library add it for everyone. Update this glob then.
publish-imdb: check-hf-auth
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

# ECG heartbeats: PTB from the private repo raoulg/mads-exam-24, MIT-BIH cut from the
# PhysioNet records and split by patient. No encoder: the 187 steps of a beat go into the
# cache as a (187, 1) sequence. Both sources are fetched on first use.
HEARTBEAT_SOURCE = artefacts/sources/mads-exam-24
MITDB_SOURCE = artefacts/sources/mitdb

$(HEARTBEAT_SOURCE):
	gh repo clone raoulg/mads-exam-24 $(HEARTBEAT_SOURCE)

$(MITDB_SOURCE):
	uv run --with wfdb python -c "import wfdb; wfdb.dl_database('mitdb', '$(MITDB_SOURCE)')"

build-heartbeats: | $(HEARTBEAT_SOURCE) $(MITDB_SOURCE)
	uv run --with wfdb python scripts/build_heartbeat_cache.py

publish-heartbeats: check-hf-auth
	@for ds in ptb mitbih; do \
		echo "==> publishing heartbeats-$$ds"; \
		uv run python scripts/publish_cache.py $(ARGS) \
			--dataset heartbeats-$$ds --card scripts/cards/heartbeats-$$ds.md \
			--train $$(ls -dt artefacts/*_heartbeats-$${ds}_train_raw_beat 2>/dev/null | head -1) \
			--test  $$(ls -dt artefacts/*_heartbeats-$${ds}_test_raw_beat 2>/dev/null | head -1) || exit 1; \
	done

publish-eurosat: check-hf-auth
	@for enc in $(ENCODERS); do \
		slug=$${enc##*/}; \
		echo "==> publishing eurosat x $$slug"; \
		uv run python scripts/publish_cache.py $(ARGS) \
			--dataset eurosat --source tanganke/eurosat \
			--train $$(ls -dt artefacts/*_eurosat_train_$${slug}_embed 2>/dev/null | head -1) \
			--test  $$(ls -dt artefacts/*_eurosat_test_$${slug}_embed 2>/dev/null | head -1) || exit 1; \
	done

publish-dtd: check-hf-auth
	@for enc in $(ENCODERS); do \
		slug=$${enc##*/}; \
		echo "==> publishing dtd x $$slug"; \
		uv run python scripts/publish_cache.py $(ARGS) \
			--dataset dtd --source tanganke/dtd \
			--train $$(ls -dt artefacts/*_dtd_train_$${slug}_embed 2>/dev/null | head -1) \
			--test  $$(ls -dt artefacts/*_dtd_test_$${slug}_embed 2>/dev/null | head -1) || exit 1; \
	done
