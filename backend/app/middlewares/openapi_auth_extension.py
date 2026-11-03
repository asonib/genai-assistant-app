from drf_spectacular.extensions import OpenApiAuthenticationExtension


class FilterPassthroughAuthenticationScheme(OpenApiAuthenticationExtension):
    """Documents the middleware token auth in Swagger/ReDoc (Authorize button)."""
    target_class = 'app.middlewares.drf_passthrough_auth.FilterPassthroughAuthentication'
    name = 'TokenAuth'

    def get_security_definition(self, auto_schema):
        return {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization',
            'description': 'Format: Bearer <token> (or Token <token>)',
        }
