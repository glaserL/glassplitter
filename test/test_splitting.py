import pytest
from glassplitter.tokenizer import Tokenizer


@pytest.fixture(
    params=[
        {"doc_type": "pdf"},
        {"doc_type": None},
    ],
    ids=["pdf", "unknown"],
)
def tokenizer(request):
    return Tokenizer(**request.param)


def test_empty_first_word(tokenizer: Tokenizer):
    spans = ["", "This is a fun sentence!"]
    result = tokenizer._split(spans)
    assert len(result) == 1


def assert_tokens_in_spans(actual, example):
    for a in actual:
        for text, span in a:
            assert text in example[span]


def test_splitting(tokenizer: Tokenizer):
    example = [
        "I am a sentence with trailing space. ",
        "Different Box",
        " from the rest of the sentence.Hello there.",
        "StudNr something wrt TUD & KHG",
    ]
    example_input = [(text, {"some_data": "schwund is immer"}) for text in example]
    actual = tokenizer.split(example_input, trim=True)
    assert len(actual) == 4, "There should be 4 sentences."
    assert_tokens_in_spans(actual, example)


def test_filtered_spans_arent_referenced(tokenizer: Tokenizer):
    spans = ["", "This is the second sentence!"]
    result = tokenizer._split(spans)
    assert all(0 != span_id for _, span_id in result[0])


def test_ids_are_assigned_correctly(tokenizer: Tokenizer):
    spans = [("I am a ", {"line": 1}), ("split sentence!", {"line": 2})]

    split = tokenizer.split(spans, trim=False)
    assert split == [
        [
            ("", 0),
            ("I", 0),
            ("am", 0),
            ("a", 0),
            ("split", 1),
            ("sentence", 1),
            ("!", 1),
            ("", 1),
        ]
    ]


def test_trim_removes_surrounding_empty_tokens(tokenizer: Tokenizer):
    spans = [("I am a ", {"line": 1}), ("split sentence!", {"line": 2})]

    split = tokenizer.split(spans, trim=True)
    assert split == [
        [
            ("I", 0),
            ("am", 0),
            ("a", 0),
            ("split", 1),
            ("sentence", 1),
            ("!", 1),
        ]
    ]
