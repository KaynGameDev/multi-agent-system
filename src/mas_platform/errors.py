class MasPlatformError(Exception):
    """Base platform error."""


class RegistryError(MasPlatformError):
    """Registry discovery or consistency failure."""


class LoadError(MasPlatformError):
    """Manifest entrypoint loading failure."""


class ValidationError(MasPlatformError):
    """Package validation failure."""

