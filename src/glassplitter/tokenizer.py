from typing import Dict, Iterable, List, Tuple

import logging

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


class Tokenizer:
    @classmethod
    def _init_tokenizer(cls):
        if not hasattr(cls, "_tok"):
            cls._tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def __init__(self, lang="en", clean=True, doc_type="pdf", join_char=" 分 ") -> None:
        self._init_tokenizer()
        segmenter = pysbd.Segmenter(language=lang, clean=clean, doc_type=doc_type)
        self._segment = lambda raw_text: segmenter.segment(raw_text)
        self._join_char = join_char

    def split(self, spans: Iterable[Tuple[str, Dict]], trim=True):
        return self._split([span for span, _ in spans], trim)

    def _split(self, spans: Iterable[str], trim=True):
        if trim:
            spans = {i: span for i, span in enumerate(spans) if len(span.strip())}
        else:
            spans = {i: span for i, span in enumerate(spans)}
        text = self._join_char.join([span for span in spans.values()])

        sents = self._segment(text)

        results: List[List[Tuple[str, int]]] = []
        span_id = min(k for k in spans.keys())
        hits, words, text_offset = 0, 0, 0

        for sent in sents:
            result: List[Tuple[str, int]] = []
            encodings = self._tok(sent, return_offsets_mapping=True)

            for i, _ in enumerate(encodings["input_ids"]):
                begin, end = encodings["offset_mapping"][i]
                token = sent[begin:end]
                if self._join_char.strip() in token:
                    span_id = next_index(span_id, spans)
                    continue

                if not trim or len(token):
                    result.append((token, span_id))

                if len(token) > 1:
                    words += 1
                    if token in spans[span_id]:
                        hits += 1

            text_offset += len(sent)

            results.append(result)

        logger.debug(f"Split {len(sents)}. {hits}/{words} hits. ({hits/words:.0%})")

        return results

    def split_flat(self, sentences: Iterable[str], trim=True):
        if trim:
            sentences = [s for s in sentences if len(s.strip())]

        text = " ".join(sentences)
        sents = self._segment(text)

        result = []
        for sent in sents:
            encodings = self._tok(sent, return_offsets_mapping=True)
            for i, _ in enumerate(encodings["input_ids"]):
                begin, end = encodings["offset_mapping"][i]
                token = sent[begin:end]
                if len(token):
                    result.append(token)

        return result
