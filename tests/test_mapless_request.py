from types import SimpleNamespace

import pytest

from limo_ros_mcp.server import LimoMCPService, build_mcp_server


class Gateway:
    def __init__(self):
        self.calls = []

    async def request_exploration(self, **kwargs):
        self.calls.append(kwargs)
        return {"state": "FINISHED", "command_dispatched": False}

    def prepare_operator_action(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(approval_request={"action_id": kwargs["action_id"]})


@pytest.mark.asyncio
async def test_shadow_exploration_submits_mapless_contract_to_daemon():
    gateway = Gateway()
    result = await LimoMCPService(gateway=gateway).request_exploration(
        duration_sec=10,
        body_snapshot_hash="a" * 64,
        execution_mode="SHADOW",
        action_id="test",
        wait_timeout_sec=1,
    )
    assert result["command_dispatched"] is False
    assert gateway.calls[0]["capability_id"] == "limo.explore"
    assert gateway.calls[0]["arguments"] == {
        "schema_version": "limo.explore.v1",
        "duration_sec": 10.0,
    }


def test_real_proposal_is_bounded_and_requires_confirmation():
    gateway = Gateway()
    service = LimoMCPService(gateway=gateway)
    result = service.prepare_exploration_confirmation(
        duration_sec=10,
        body_snapshot_hash="a" * 64,
        action_id="test",
        deadline_at="2026-11-18T10:00:00Z",
    )
    assert result["decision"] == "CONFIRMATION_REQUIRED"
    assert gateway.calls[0]["capability_id"] == "limo.explore"
    assert result["command_dispatched"] is False
    invalid = service.prepare_exploration_confirmation(
        duration_sec=31,
        body_snapshot_hash="a" * 64,
        action_id="bad",
        deadline_at="2026-11-18T10:00:00Z",
    )
    assert invalid["ok"] is False
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
async def test_inspection_profile_never_exposes_exploration_control():
    server = build_mcp_server(LimoMCPService(gateway=Gateway()), profile="inspection")
    assert "limo_request_exploration" not in {tool.name for tool in await server.list_tools()}
