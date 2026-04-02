"""Application bootstrap and shared runtime modules for Jade Agent."""

__all__ = [
    "Settings",
    "build_agent_graph",
    "build_web_agent_registrations",
    "load_settings",
]


def __getattr__(name: str):
    if name in {"Settings", "load_settings"}:
        from app.config import Settings, load_settings

        exports = {
            "Settings": Settings,
            "load_settings": load_settings,
        }
        return exports[name]
    if name in {"build_agent_graph", "build_web_agent_registrations"}:
        from app.graph import build_agent_graph, build_web_agent_registrations

        exports = {
            "build_agent_graph": build_agent_graph,
            "build_web_agent_registrations": build_web_agent_registrations,
        }
        return exports[name]
    raise AttributeError(f"module 'app' has no attribute {name!r}")
