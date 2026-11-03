from rest_framework.pagination import PageNumberPagination

from app.config.app_config import AppConfigConstants


class StandardResultsPagination(PageNumberPagination):
    """Shared pagination: ?page=<n>&page_size=<m> -> {count, next, previous, results}."""
    page_size = AppConfigConstants.DEFAULT_PAGE_SIZE
    page_size_query_param = 'page_size'
    max_page_size = AppConfigConstants.MAX_PAGE_SIZE
