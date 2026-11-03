from django.urls import path, re_path
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView

from app.controllers.auth_controller import LoginController, LogoutController, MeController, RegisterController
from app.controllers.chat_controller import (
    ChatController,
    ConversationDetailController,
    ConversationListController,
    ToolListController,
)
from app.controllers.document_controller import (
    DocumentDetailController,
    DocumentListCreateController,
    DocumentReingestController,
)
from app.controllers.health_controller import HealthController

# Trailing slashes are optional everywhere (`/?$`) so POSTs never hit APPEND_SLASH redirects.
urlpatterns = [
    path('schema/', SpectacularAPIView.as_view(), name='schema'),
    path('docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),

    re_path(r'^health/?$', HealthController.as_view(), name='health'),

    re_path(r'^auth/register/?$', RegisterController.as_view(), name='auth-register'),
    re_path(r'^auth/login/?$', LoginController.as_view(), name='auth-login'),
    re_path(r'^auth/logout/?$', LogoutController.as_view(), name='auth-logout'),
    re_path(r'^auth/me/?$', MeController.as_view(), name='auth-me'),

    re_path(r'^documents/?$', DocumentListCreateController.as_view(), name='document-list'),
    re_path(r'^documents/(?P<pk>\d+)/?$', DocumentDetailController.as_view(), name='document-detail'),
    re_path(r'^documents/(?P<pk>\d+)/reingest/?$', DocumentReingestController.as_view(), name='document-reingest'),

    re_path(r'^chat/?$', ChatController.as_view(), name='chat'),
    re_path(r'^conversations/?$', ConversationListController.as_view(), name='conversation-list'),
    re_path(r'^conversations/(?P<pk>\d+)/?$', ConversationDetailController.as_view(), name='conversation-detail'),
    re_path(r'^tools/?$', ToolListController.as_view(), name='tool-list'),
]
