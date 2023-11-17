import logging
from typing import Dict, Iterable, List, Tuple

import pysbd
from transformers import BertTokenizerFast

logger = logging.getLogger(__name__)


class Tokenizer:
    @classmethod
    def _init_tokenizer(cls):
        if not hasattr(cls, "_tok"):
            cls._tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def __init__(self, lang="en", clean=True, doc_type="None") -> None:
        self._init_tokenizer()
        self.segmenter = pysbd.Segmenter(language=lang, clean=clean, doc_type=doc_type)

    def split(self, spans: Iterable[Tuple[str, Dict]]):
        spans = [span for span, _ in spans]
        text = " ".join([span for span in spans])
        cum_lens = [
            sum(len(prev) for prev in spans[0 : i + 1]) for i in range(len(spans))
        ]

        sentences = self.segmenter.segment(text)

        results: List[List[Tuple[str, int]]] = []
        span_id = 0
        hits = 0
        total = 0
        text_index = 0
        for sent in sentences:
            result: List[Tuple[str, int]] = []
            encodings = self._tok(sent, return_offsets_mapping=True)

            for i, _ in enumerate(encodings["input_ids"]):
                begin, end = encodings["offset_mapping"][i]
                token = sent[begin:end]
                total += 1

                text_index += len(token)
                result.append((token, span_id))
                soundness = token in spans[span_id]
                if soundness:
                    hits += 1

                if text_index > cum_lens[span_id]:
                    span_id += 1

            results.append(result)

        logger.debug(
            f"Split into {len(sentences)}. {hits}/{total} hits. ({hits/total:.0%})"
        )

        return results
