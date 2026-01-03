from collections import deque
from typing import Deque, List, Tuple


class ConversationHistory:
    """Rolling in-memory history of user/assistant turns."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._messages: Deque[Tuple[str, str]] = deque(maxlen=max_turns)

    def add_turn(self, user: str, assistant: str) -> None:
        self._messages.append((user, assistant))

    def to_messages(self) -> List[dict]:
        messages: List[dict] = []
        for user_text, assistant_text in self._messages:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        return messages
