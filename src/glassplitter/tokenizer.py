import logging
from typing import Dict, Iterable, List, Tuple

import pysbd
from transformers import BertTokenizerFast

logger = logging.getLogger(__name__)


def next_index(current_index: int, span_dict: Dict[int, str]):
    end = max(span_dict.keys())
    start = min(current_index + 1, end)
    for candidate in range(start, end):
        if candidate in span_dict.keys():
            return candidate
    return end


def cumulative_lengths(span_dict: Dict[int, str]) -> Dict[int, int]:
    result = {}
    for i in span_dict.keys():
        cumulative = sum(len(span_dict.get(j, "")) for j in range(0, i + 1))
        result[i] = cumulative
    return result


class Tokenizer:
    @classmethod
    def _init_tokenizer(cls):
        if not hasattr(cls, "_tok"):
            cls._tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def __init__(self, lang="en", clean=True, doc_type="pdf") -> None:
        self._init_tokenizer()
        self.segmenter = pysbd.Segmenter(language=lang, clean=clean, doc_type=doc_type)

    def split(self, spans: Iterable[Tuple[str, Dict]], trim=True):
        return self._split([span for span, _ in spans], trim)

    def _split(self, spans: Iterable[str], trim=True):
        if trim:
            spans = {i: span for i, span in enumerate(spans) if len(span.strip())}
        else:
            spans = {i: span for i, span in enumerate(spans)}
        text = " ".join([span for span in spans.values()])
        cum_lens = cumulative_lengths(spans)

        sents = self.segmenter.segment(text)

        results: List[List[Tuple[str, int]]] = []
        span_id = min(k for k in spans.keys())
        hits, words, text_offset = 0, 0, 0

        for sent in sents:
            result: List[Tuple[str, int]] = []
            encodings = self._tok(sent, return_offsets_mapping=True)

            for i, _ in enumerate(encodings["input_ids"]):
                begin, end = encodings["offset_mapping"][i]
                token = sent[begin:end]

                j = next_index(span_id, spans)

                if (text_offset + begin) >= cum_lens[span_id]:
                    span_id = j

                result.append((token, span_id))

                if len(token) > 1:
                    words += 1
                    if token in spans[span_id]:
                        hits += 1

            text_offset += len(sent)

            results.append(result)

        logger.debug(f"Split {len(sents)}. {hits}/{words} hits. ({hits/words:.0%})")

        return results
