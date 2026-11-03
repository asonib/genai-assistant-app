from abc import ABC, abstractmethod


class IChatService(ABC):
    @abstractmethod
    def send_message(self, user, message, conversation_id=None, document_ids=None):
        """Run one agent turn. Returns (conversation, user_message, assistant_message)."""
        pass

    @abstractmethod
    def list_available_tools(self) -> list:
        pass
