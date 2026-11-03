from abc import ABC, abstractmethod


class IAuthService(ABC):
    @abstractmethod
    def register_user(self, username, password, email=None, user_agent=''):
        """Returns (user, raw_token, session)."""
        pass

    @abstractmethod
    def login_user(self, username, password, user_agent=''):
        """Returns (user, raw_token, session)."""
        pass

    @abstractmethod
    def logout(self, session, all_sessions=False) -> int:
        pass
