"""Gateway routing package."""

from gateway.agent import GatewayNode, build_gateway_prompt

__all__ = [
    "GatewayNode",
    "build_gateway_prompt",
]
