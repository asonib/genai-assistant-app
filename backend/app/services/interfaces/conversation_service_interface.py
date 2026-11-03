from abc import ABC, abstractmethod


class IConversationService(ABC):
    @abstractmethod
    def list_conversations(self, user):
        pass

    @abstractmethod
    def get_conversation(self, user, conversation_id):
        pass

    @abstractmethod
    def create_conversation(self, user, title):
        pass

    @abstractmethod
    def rename_conversation(self, user, conversation_id, title):
        pass

    @abstractmethod
    def delete_conversation(self, user, conversation_id) -> None:
        pass

    @abstractmethod
    def get_memory(self, conversation) -> list:
        """Recent turns as [{"role", "content"}], oldest first, bounded by AIConfigConstants."""
        pass
