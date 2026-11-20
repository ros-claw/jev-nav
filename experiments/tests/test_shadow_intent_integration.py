"""Bridge an advisory decision into a synthetic ROSClaw executor, without I/O."""
from pathlib import Path
import sys
import unittest

import shadow_intent


@unittest.skipIf(sys.version_info < (3, 11), 'ROSClaw requires Python 3.11+')
class ShadowIntentIntegrationTests(unittest.TestCase):
    def test_distinct_model_advice_rounds_count_in_synthetic_mission(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'worktrees/rosclaw/src'))
        from rosclaw.kernel import ActionEnvelope, ExecutionMode
        from rosclaw.robot_pack.limo_shadow_mission import ShadowMissionBudget

        body_hash = 'sha256:' + 'a' * 64
        budget = ShadowMissionBudget(body_id='limo', body_snapshot_hash=body_hash,
            session_id='fixture', generation='synthetic-boot', started_at=100,
            expires_at=110, max_steps=3, max_duration_sec=1.5)
        decision = {'status': 'ok', 'choice': 'forward', 'executable': False,
                    'round_id': 'round-1', 'observation_hash': 'sha256:' + 'b' * 64,
                    'expires_wall_time': 202.0}

        def make_action(advice, action_id, mono):
            request = shadow_intent.make_intent(advice, action_id, body_hash,
                wall_now=200, monotonic_now=mono, duration=.25)
            return ActionEnvelope(action_id=action_id, actor_id='fixture', agent_framework='unittest',
                session_id='fixture', body_id='limo', body_snapshot_hash=body_hash,
                capability_id='limo.mapless_step', execution_mode=ExecutionMode.SHADOW, arguments=request)

        first = make_action(decision, 'action-1', 100.0)
        replay = make_action(decision, 'action-2', 100.1)
        self.assertTrue(budget.accept(first, generation='synthetic-boot', now=100.1)['accepted'])
        self.assertEqual(budget.accept(replay, generation='synthetic-boot', now=100.1)['reason'],
                         'observation_reused')
        decision.update(round_id='round-2', observation_hash='sha256:' + 'c' * 64, choice='left')
        second = make_action(decision, 'action-3', 100.2)
        self.assertTrue(budget.accept(second, generation='synthetic-boot', now=100.2)['accepted'])
        self.assertFalse(budget.status()['hardware_authorized'])

    def test_decision_is_accepted_only_while_bound_and_fresh(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'worktrees/rosclaw/src'))
        from rosclaw.kernel import ActionEnvelope, ActionState, ExecutionMode
        from rosclaw.robot_pack.limo_mapless import MaplessShadowExecutor

        body_hash = 'sha256:' + 'a' * 64
        decision = {'status': 'ok', 'choice': 'forward', 'executable': False,
                    'round_id': 'round-1', 'observation_hash': 'sha256:' + 'b' * 64,
                    'expires_wall_time': 102.0}
        request = shadow_intent.make_intent(decision, 'action-1', body_hash,
                                            wall_now=100, monotonic_now=50, duration=.25)
        action = ActionEnvelope(action_id='action-1', actor_id='fixture', agent_framework='unittest',
            session_id='fixture', body_id='limo', body_snapshot_hash=body_hash,
            capability_id='limo.mapless_step', execution_mode=ExecutionMode.SHADOW, arguments=request)
        accepted = MaplessShadowExecutor(body_id='limo', body_snapshot_hash=body_hash,
                                          clock=lambda: 50.1)(action)
        expired = MaplessShadowExecutor(body_id='limo', body_snapshot_hash=body_hash,
                                         clock=lambda: 51.0)(action)
        self.assertEqual(accepted.final_state, ActionState.COMPLETED)
        self.assertFalse(accepted.dispatch_result['hardware_dispatched'])
        self.assertEqual(expired.final_state, ActionState.BLOCKED)
        self.assertIsNone(expired.driver_ack)
