from rest_framework import serializers


class RegisterSerializer(serializers.Serializer):
    username = serializers.RegexField(r'^[\w.@+-]+$', max_length=150, min_length=3)
    password = serializers.CharField(write_only=True, min_length=8, max_length=128)
    email = serializers.EmailField(required=False, allow_blank=True)


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True, max_length=128)


class LogoutSerializer(serializers.Serializer):
    all_sessions = serializers.BooleanField(required=False, default=False)


class AuthTokenResponseSerializer(serializers.Serializer):
    token = serializers.CharField()
    token_type = serializers.CharField()
    expires_at = serializers.DateTimeField()
    user = serializers.DictField()


class MeResponseSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    username = serializers.CharField()
    email = serializers.CharField()
    session_expires_at = serializers.DateTimeField()
