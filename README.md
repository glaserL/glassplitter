# glassplitter

A sentence splitter and tokenizer.

Special focus is put on splitting and tokenizing spans of texts from pdfs.
For downstream applications we want to retain a pointer span based metadata for each token.

# Usage

```python
from glassplitter import Tokenizer

spans = [("I am a ", {"line": 1}), ("split sentence!", {"line": 2})]
tokenizer = Tokenizer(lang="en", clean=True, doc_type="pdf")

split = tokenizer.split(spans, trim=True)
# [[("", 0), ("I", 0), ("am", 0), ("a", 0), ("split", 1), ("sentence", 1), ("!", 1), ("", 1)]]
```
