# Papers

One anchor paper per lesson, chosen so the paper's *method* is the lesson's method rather
than merely its topic. Every arXiv id below was checked against arXiv, not recalled.

`✔` means a PDF is already in `references/`. The rest are open-access links — download
what you want locally, but **do not commit PDFs of paywalled work**, and keep in mind that
anything in this folder can end up in a build artefact.

| # | lesson | paper | link | local |
|---|---|---|---|---|
| L1 | Representation, composition, search | Razavian, Azizpour, Sullivan & Carlsson (2014), *CNN Features Off-the-Shelf: An Astounding Baseline for Recognition* | [arXiv:1403.6382](https://arxiv.org/abs/1403.6382) | |
| L2 | Time series | Oreshkin et al. (2020), *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting* | [arXiv:1905.10437](https://arxiv.org/abs/1905.10437) | |
| L2 | — paired with | Zeng, Chen, Zhang & Xu (2023), *Are Transformers Effective for Time Series Forecasting?* | [arXiv:2205.13504](https://arxiv.org/abs/2205.13504) | |
| L3 | Scaling and search | Kaplan et al. (2020), *Scaling Laws for Neural Language Models* | [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) | ✔ |
| L3 | — paired with | Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* (Chinchilla) | [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) | |
| L3 | — notebook 01 | Srivastava, Greff & Schmidhuber (2015), *Highway Networks* | [arXiv:1505.00387](https://arxiv.org/abs/1505.00387) | ✔ |
| L3 | — notebook 01 | Shazeer et al. (2017), *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* | [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) | ✔ |
| L3 | — notebook 03 | Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization*, JMLR 13 | [jmlr.org](https://www.jmlr.org/papers/v13/bergstra12a.html) | |
| L4 | Language | Vaswani et al. (2017), *Attention Is All You Need* | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | ✔ |
| L5 | Self-supervised | Assran et al. (2023), *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA) | [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) | |
| L5 | — paired with | Chen, Kornblith, Norouzi & Hinton (2020), *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR) | [arXiv:2002.05709](https://arxiv.org/abs/2002.05709) | |
| L6 | Uncertainty | Angelopoulos & Bates (2021), *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification* | [arXiv:2107.07511](https://arxiv.org/abs/2107.07511) | |
| L7 | Generalists and evaluation | Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) | |

## Why these, and why some come in pairs

**L1 — Razavian et al. (2014).** The entire thesis of this course, a decade early and
stated as a surprise rather than as received wisdom: take a network trained for something
else, freeze it, train a linear classifier on its features, and beat purpose-built
pipelines across object classification, scene recognition, fine-grained recognition and
retrieval. Students meet "the representation is the model" as the claim that it was.

**L2 — N-BEATS against DLinear.** N-BEATS is the architecture the lesson builds: a basis
expansion that constrains what the model may say, which is what makes it interpretable.
Zeng et al. is the paper showing a single linear layer beating elaborate transformer
forecasters on the standard benchmarks. **The pairing is the lesson** — students build the
sophisticated thing and measure it against the trivial thing, which is what the notebooks
make them do anyway.

**L3 — Kaplan against Chinchilla.** Kaplan et al. established smooth power laws and
concluded that extra compute should mostly buy *parameters*. That shaped how models were
built for two years. Hoffmann et al. re-ran it varying data and parameters together
instead of holding one nearly fixed, and found models had been far too large for their
data — a 70B model beating a 280B one. **A widely believed empirical law was wrong because
of how the experiment was designed, and the fix required no new idea, only a more careful
measurement.** That is this course's standard, at a scale of millions of dollars, and it
rhymes with what `02_scaling_curves` measures on a laptop: data is the axis that moves.

**L4 — Vaswani et al.** The mechanism `01_attention` derives by hand before importing it.

**L5 — I-JEPA with SimCLR.** The two pretext designs the lesson implements, and the
clearest available statement of "predict in latent space, not in pixels" — which is the
distinction `01_why_pretext_tasks_work` sets up by measuring what a reconstruction loss
chooses to keep.

**L6 — Angelopoulos & Bates.** Coverage guarantees with no distributional assumption and
no retraining, applied to a model you already have. Unusually readable for what it does.

**L7 — Zheng et al.** The paper that popularised LLM-as-a-judge *and* measured its failure
modes — position bias, verbosity bias, self-enhancement bias, limited reasoning. Notebook
02 reproduces the position-bias check at laptop scale, which is the whole argument for
validating a judge before trusting it.

## The pattern

Note that L2 and L3 are pairings in which **the second paper deflates the first**, and
that L7's single paper deflates its own technique. That is not an accident of selection —
it is the shape of the habit this course grades. A student who can build N-BEATS is
useful; a student who builds it, measures it against a linear layer, and reports honestly
which won is the one this course is trying to produce.
