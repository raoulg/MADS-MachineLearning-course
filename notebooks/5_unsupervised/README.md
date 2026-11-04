# Lesson 5 — Self-supervised: where the representation comes from

Lesson 1 handed you an encoder and showed it dominates every other choice: the encoder
moved accuracy 10 to 40 points, the head architecture less than one. It never said where
such an encoder comes from. That is this lesson.

| notebook | question |
|---|---|
| `01_why_pretext_tasks_work` | Why does training on a task you do not care about produce a representation for tasks you do? |
| `02_selfsupervised` | Two views of one image, the collapse that idea walks into, and the block that prevents it |
| `03_patch_tokens` | What pooling throws away, and when you should keep the patch axis |
| `04_ijepa` | Predicting what you were not shown — and a loss that lies |

## The instrument

Every claim in this lesson is measured with a **linear probe**: freeze the encoder, fit a
linear classifier on its output, report accuracy. Nothing else is trained.

The probe is the right instrument *because* it is weak. A deep head could recover class
structure from almost any representation given enough capacity, and then you would be
measuring the head. A linear probe can only read structure that is already laid out
linearly, so it asks the question that matters: did the encoder do the work?

The second instrument is a **collapse detector** — the mean per-dimension standard
deviation of L2-normalised embeddings. Self-supervised losses have a trivial optimum
(output the same vector for every input), the loss curve looks *excellent* while it
happens, and without this instrument you will not notice. `02_selfsupervised` walks into
that trap deliberately before showing the fix.

## The through-line

The pretext task is not a trick for getting labels for free. It is the choice of **what
the representation is allowed to forget**:

- reconstruction forgets almost nothing, so it keeps brightness and texture (`01`)
- classification forgets everything except the boundaries its labels named (`01`)
- contrastive learning forgets whatever your augmentation changes — which makes the
  augmentation a design decision you can get wrong, and measure (`02`)
- masked prediction in latent space forgets whatever is predictable from context (`04`)

## Running

Everything runs on CPU from the notebooks themselves; there is nothing to build first.
Expensive constants use `param(full, test=...)`, so the same notebooks run fast under
`notebooktester` and at full size for you.

```bash
uv run notebooktester notebooks/5_unsupervised -t 300 -pb
```
