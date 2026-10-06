import asyncio

import pytest
from fastmcp import Client

from voicesmith import mcp_server, voices


def run(coro):
    return asyncio.run(coro)


async def _tools():
    async with Client(mcp_server.server) as c:
        return [t.name for t in await c.list_tools()]


def test_agents_get_no_tool_that_can_grant_consent():
    names = run(_tools())
    assert {"list_voices", "speak", "job_status", "verify_audio", "consent_steps", "doctor"} <= set(names)
    forbidden = [n for n in names if any(w in n for w in ("attest", "verify_consent", "grant", "approve", "record"))]
    assert forbidden == []


def test_speak_refuses_a_voice_without_consent():
    voices.create("dana", "Dana Example")

    async def go():
        async with Client(mcp_server.server) as c:
            return await c.call_tool("speak", {"voice": "dana", "text": "hello"}, raise_on_error=False)

    res = run(go())
    assert res.is_error
    assert "consent" in str(res.content).lower()


def test_list_voices_and_consent_steps():
    voices.create("erin", "Erin Example")

    async def go():
        async with Client(mcp_server.server) as c:
            listed = await c.call_tool("list_voices", {})
            steps = await c.call_tool("consent_steps", {"name": "erin"})
            return listed.data, steps.data

    listed, steps = run(go())
    assert listed[0]["name"] == "erin" and listed[0]["ready"] is False
    assert any("consent request erin" in s for s in steps["steps"])


@pytest.mark.parametrize("name", ["../escape", "UPPER", ""])
def test_bad_voice_names_are_rejected(name):
    with pytest.raises(ValueError):
        voices.create(name, "X")
