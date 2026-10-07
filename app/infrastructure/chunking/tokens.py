"""Token counting for chunk sizing.

Models limit and bill by *tokens*, not characters (1 token ≈ 4 characters of
English). We count with tiktoken's `o200k_base` encoding. It is not the exact
Qwen tokenizer, but chunk sizes only need to be approximately right, and it
is fast and dependency-light.
"""

import tiktoken


class TiktokenCounter:
    def __init__(self, encoding_name: str = "o200k_base") -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def split(self, text: str, max_tokens: int) -> list[str]:
        tokens = self._encoding.encode(text, disallowed_special=())
        return [
            self._encoding.decode(tokens[i : i + max_tokens])
            for i in range(0, len(tokens), max_tokens)
        ]
