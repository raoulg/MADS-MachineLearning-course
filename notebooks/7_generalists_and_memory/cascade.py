"""embedding -> router -> { head | LLM }, wrapped as one callable.

The router and the head are both `vectormesh` pipelines; the LLM call is neither. An HTTP
request against text cannot sit inside a `Serial` the way `MoE` holds its experts, so the
split between the two branches is a plain conditional, not a fused module -- `Cascade` is
that conditional, wrapped so a caller sees one `.predict()` rather than the pipeline underneath.
"""

from __future__ import annotations

import numpy as np
import torch

from head import make_loader, predict


class Cascade:
    """Route each document to the head or the LLM, then predict on whichever it lands on.

    `router` and `head` are trained `vectormesh` pipelines over the same `(chunks, dim)`
    embeddings, and may use different `max_chunks`. `llm` is a `CachedLLM`. `tau` is the
    router's decision threshold on P(the head owns this document) -- not the same tau as
    the head-profile's recall threshold, though the two are related: it is what the router
    learned to reproduce.

    `dataset` needs a `onehot` column for `Collate` to batch, even though `Cascade` never
    reads it -- a genuinely label-free `Cascade.predict` is one collate function away
    (E1 in `04`), not built here for lack of a use for it in this lesson.
    """

    def __init__(
        self,
        router,
        head,
        llm,
        areas: tuple[str, ...],
        router_max_chunks: int,
        head_max_chunks: int,
        tau: float = 0.5,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.router = router.eval()
        self.head = head.eval()
        self.llm = llm
        self.areas = areas
        self.router_max_chunks = router_max_chunks
        self.head_max_chunks = head_max_chunks
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

    def route(self, dataset) -> np.ndarray:
        """True where the router sends a document to the head, False for the LLM."""
        loader = make_loader(
            dataset, self.router_max_chunks, shuffle=False, batch_size=self.batch_size
        )
        scores = []
        with torch.no_grad():
            for X, _ in loader:
                p_owned = torch.softmax(self.router(X.to(self.device)), dim=-1)[:, 1]
                scores.append(p_owned.cpu())
        return torch.cat(scores).numpy() >= self.tau

    def predict(self, dataset, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Predicted label index per row of `dataset`, plus which branch it took.

        `texts` must be the opinion behind each row, in the same order -- the head never
        touches it, but the LLM branch has nothing else to read. -1 marks an LLM answer
        that could not be parsed into a label.
        """
        owned = self.route(dataset)

        head_pred, _ = predict(
            self.head,
            make_loader(
                dataset, self.head_max_chunks, shuffle=False, batch_size=self.batch_size
            ),
            self.device,
        )
        out = head_pred.copy()

        escalated = np.flatnonzero(~owned)
        if len(escalated):
            llm_texts = [texts[i] for i in escalated]
            llm_pred = self.llm.classify(llm_texts, self.areas)
            out[escalated] = [p if p is not None else -1 for p in llm_pred]

        return out, owned
