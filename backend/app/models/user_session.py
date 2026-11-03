import hashlib
import secrets
from datetime import timedelta

from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone

from app.config.app_config import AppConfigConstants


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode('utf-8')).hexdigest()


class UserSession(models.Model):
    """
    An API login session. Only the SHA-256 of the bearer token is stored, so a
    leaked database does not leak usable tokens. The raw token is returned to
    the client exactly once (on login/register).
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    token_hash = models.CharField(max_length=64, unique=True, db_index=True)
    user_agent = models.CharField(max_length=255, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    last_seen_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()

    @classmethod
    def create_session(cls, user, user_agent: str = ''):
        """Returns (session, raw_token)."""
        raw_token = secrets.token_hex(32)
        session = cls.objects.create(
            user=user,
            token_hash=hash_token(raw_token),
            user_agent=(user_agent or '')[:255],
            expires_at=timezone.now() + timedelta(days=AppConfigConstants.TOKEN_EXPIRY_DAYS),
        )
        return session, raw_token

    @classmethod
    def find_by_raw_token(cls, raw_token: str):
        return cls.objects.select_related('user').filter(token_hash=hash_token(raw_token)).first()

    def is_expired(self):
        return timezone.now() > self.expires_at

    def __str__(self):
        return f"Session({self.user.username}, expires_at={self.expires_at})"
