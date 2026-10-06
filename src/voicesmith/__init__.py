"""voicesmith: local, consent-first voice cloning.

Keep this module import-light. The CLI, MCP server and tests import submodules
directly, and engine workers never import this package at all.
"""

__version__ = "0.1.0"
