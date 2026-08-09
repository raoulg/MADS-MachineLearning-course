"""Ask an LLM for predictions — from a published cache, or live from your own ollama.

Every graded cell in this lesson runs off the cache, so the notebooks work on a laptop with
no GPU and no ollama, and give everyone the same numbers. If you *have* the hardware, build
the client with `live=True` and the same call regenerates the answers on your machine.

    llm = CachedLLM(DATA / "scotus_llm_predictions.json")               # instant
    llm = CachedLLM(DATA / "scotus_llm_predictions.json", live=True)    # your ollama
    preds = llm.classify(texts, AREAS)

That switch is the cache pattern from lesson 1 applied to generation: the expensive step
happens once, on hardware that has a GPU, and everyone else gets to spend their time on the
thinking instead.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


class CachedLLM:
    """A classifier backed by a JSON file of answers, refilled from ollama on demand.

    The cache file is named when you build the client, so two clients over two files (a
    published one and your own) can coexist without either reaching for a module global.
    """

    def __init__(
        self,
        cache_path: Path | str,
        model: str = "qwen3.6:35b",
        live: bool = False,
        url: str = OLLAMA_URL,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.model = model
        self.live = live
        self.url = url
        self.store: dict[str, dict] = (
            json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        )

    def __len__(self) -> int:
        return len(self.store)

    @staticmethod
    def _key(text: str, labels: tuple[str, ...]) -> str:
        """Content hash, so a cached answer is tied to the exact prompt that produced it."""
        h = hashlib.sha256()
        h.update(text.encode())
        h.update("|".join(labels).encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _prompt(text: str, labels: tuple[str, ...]) -> str:
        menu = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(labels))
        return (
            "Classify this document into one of these categories. Answer with the number "
            f"only.\n\n{menu}\n\nDocument: {text}\n\nAnswer:"
        )

    @staticmethod
    def _parse(raw: str, n_labels: int) -> int | None:
        for tok in "".join(c if c.isdigit() else " " for c in raw).split():
            v = int(tok)
            if 1 <= v <= n_labels:
                return v - 1
        return None

    def _ask(self, text: str, labels: tuple[str, ...]) -> str:
        r = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": self._prompt(text, labels),
                "stream": False,
                # otherwise the token budget goes to reasoning and the answer comes back empty
                "think": False,
                "options": {"num_predict": 8, "temperature": 0.0},
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def classify(self, texts: list[str], labels: tuple[str, ...]) -> list[int | None]:
        """Predicted label index per document; `None` where the answer could not be parsed.

        A cached client never touches the network and raises if a document is missing. A
        live one calls ollama for the misses and writes them back, so a second run is
        instant and you extend the cache rather than rebuild it.
        """
        out: list[int | None] = []
        missing = 0
        for text in texts:
            k = self._key(text, labels)
            if k in self.store:
                out.append(self.store[k]["pred"])
            elif self.live:
                raw = self._ask(text, labels)
                pred = self._parse(raw, len(labels))
                self.store[k] = {"raw": raw, "pred": pred, "model": self.model}
                out.append(pred)
            else:
                out.append(None)
                missing += 1

        if self.live:
            self.save()
        elif missing:
            raise KeyError(
                f"{missing} of {len(texts)} documents are not in {self.cache_path.name}. "
                "Either you changed the sample, or you need live=True to generate them. "
                "The cache is keyed on the exact prompt text, so a different truncation is "
                "a different key."
            )
        return out

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.store, indent=0))
