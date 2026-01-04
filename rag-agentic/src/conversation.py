from collections import deque
from typing import Deque, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage


class ConversationHistory:
    """Rolling in-memory history of user/assistant turns."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._messages: Deque[Tuple[str, str]] = deque(maxlen=max_turns)

    def add_turn(self, user: str, assistant: str) -> None:
        self._messages.append((user, assistant))

    def to_messages(self) -> List[dict]:
        """Return OpenAI-style messages for the stored turns."""
        messages: List[dict] = []
        for user_text, assistant_text in self._messages:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        return messages

    def to_langchain(self) -> List:
        """Return history as LangChain messages (HumanMessage/AIMessage)."""
        messages: List = []
        for user_text, assistant_text in self._messages:
            messages.append(HumanMessage(content=user_text))
            messages.append(AIMessage(content=assistant_text))
        return messages
