# Hermes-AI V2 Engine — Off-chain agent registry, match ledger, and leaderboard logic.
# Single-file implementation aligned with HermesAIV2.sol (Caduceus relay, tiered rewards).

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import hashlib
import json
import struct
import time

# ─── Constants (match HermesAIV2.sol) ─────────────────────────────────────────

SCALE = 10**18
FEE_BASIS = 920
FEE_DENOM = 10_000
MAX_PENDING = 48
MATCH_TIMEOUT_BLOCKS = 9600
EPOCH_BLOCKS = 302400
FLOOR_STAKE_WEI = 2_000_000_000_000_000
REGISTER_FEE_WEI = 200_000_000_000_000
VERSION = 2

CADUCEUS_NAMESPACE = "0x2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b"
GOVERNOR = "0x2a3b4c5D6e7F8A9b0C1d2E3f4A5b6C7d8E9f0A1b2"
VAULT = "0x3b4c5d6E7f8A9B0c1D2e3F4a5B6c7D8e9F0a1B2c3"
ADJUDICATOR = "0x4c5d6e7F8a9B0C1d2E3f4A5b6C7d8E9f0A1b2C3d4"
BOUNTY_POOL = "0x5d6e7f8A9b0C1D2e3F4a5B6c7D8e9F0a1B2c3D4e5"
RELAYER = "0x6e7f8a9B0c1D2E3f4A5b6C7d8E9f0A1b2C3d4E5f6"


class MatchState(Enum):
    OPEN = 0
    ACTIVE = 1
    SETTLED = 2
    CANCELLED = 3
    TIMED_OUT = 4


@dataclass
class AgentRecord:
    name_hash: str
    wins: int
    losses: int
    draws: int
    tier_points: int
    registered_block: int
    accrued_reward: int
    active: bool


@dataclass
class MatchSlot:
    match_id: int
    initiator: str
    opponent: str
    stake_wei: int
    created_block: int
    accepted_block: int
    state: MatchState
    proof_hash: str
    victor: Optional[str]


@dataclass
class EpochMeta:
    epoch_id: int
    start_block: int
    end_block: int
    match_count: int
    board_root: str
    sealed: bool


class HermesAIV2Engine:
    """Hermes-AI V2 off-chain engine: agents, matches, epochs, leaderboard."""

    def __init__(self, genesis_block: int = 0):
        self.genesis_block = genesis_block
        self._agents: Dict[str, AgentRecord] = {}
        self._matches: Dict[int, MatchSlot] = {}
        self._epochs: Dict[int, EpochMeta] = {}
        self._matches_by_initiator: Dict[str, List[int]] = {}
        self._matches_by_opponent: Dict[str, List[int]] = {}
        self._epoch_boards: Dict[int, List[str]] = {}
        self._next_match_id = 0
        self._current_epoch_id = 1
        self._total_stake_held = 0
        self._total_fees = 0
        self._total_agents = 0

        self._epochs[1] = EpochMeta(
            epoch_id=1,
            start_block=genesis_block,
            end_block=genesis_block + EPOCH_BLOCKS,
            match_count=0,
            board_root="",
            sealed=False,
        )

    def name_hash(self, name: str) -> str:
        h = hashlib.sha256((name or "").encode()).hexdigest()
        return "0x" + h.zfill(64)

    def register_agent_local(self, address: str, name: str) -> None:
        if address in self._agents:
            raise ValueError("HermesAIV2_AgentExists")
        nh = self.name_hash(name)
        self._agents[address] = AgentRecord(
            name_hash=nh,
            wins=0,
            losses=0,
            draws=0,
            tier_points=0,
            registered_block=self.genesis_block,
            accrued_reward=0,
            active=True,
        )
        self._total_agents += 1
        self._matches_by_initiator[address] = []
        self._matches_by_opponent[address] = []

    def create_match_local(self, initiator: str, opponent: str, stake_wei: int) -> int:
        if initiator not in self._agents or opponent not in self._agents:
            raise ValueError("HermesAIV2_AgentMissing")
        if initiator == opponent:
            raise ValueError("HermesAIV2_SelfMatch")
        if stake_wei < FLOOR_STAKE_WEI:
            raise ValueError("HermesAIV2_StakeTooLow")
        pending = len([m for m in self._matches_by_initiator.get(initiator, []) if self._matches[m].state == MatchState.OPEN])
        if pending >= MAX_PENDING:
            raise ValueError("HermesAIV2_PendingLimit")

        match_id = self._next_match_id
        self._next_match_id += 1
        self._matches[match_id] = MatchSlot(
            match_id=match_id,
            initiator=initiator,
            opponent=opponent,
            stake_wei=stake_wei,
            created_block=self.genesis_block,
            accepted_block=0,
            state=MatchState.OPEN,
            proof_hash="",
            victor=None,
        )
        self._matches_by_initiator.setdefault(initiator, []).append(match_id)
        self._matches_by_opponent.setdefault(opponent, []).append(match_id)
        self._total_stake_held += stake_wei
        self._epochs[self._current_epoch_id].match_count += 1
        return match_id

    def accept_match_local(self, match_id: int, current_block: int) -> None:
        if match_id not in self._matches:
            raise ValueError("HermesAIV2_MatchMissing")
        m = self._matches[match_id]
        if m.state != MatchState.OPEN:
            raise ValueError("HermesAIV2_MatchNotOpen")
        m.accepted_block = current_block
        m.state = MatchState.ACTIVE
        self._total_stake_held += m.stake_wei

    def settle_match_local(self, match_id: int, victor: Optional[str], current_block: int) -> None:
        if match_id not in self._matches:
            raise ValueError("HermesAIV2_MatchMissing")
        m = self._matches[match_id]
        if m.state != MatchState.ACTIVE:
            raise ValueError("HermesAIV2_MatchNotActive")
        if victor is not None and victor not in (m.initiator, m.opponent):
            raise ValueError("HermesAIV2_BadOutcome")

        m.state = MatchState.SETTLED
        m.victor = victor
        pool = m.stake_wei * 2
        fee = (pool * FEE_BASIS) // FEE_DENOM
        to_victor = pool - fee
        self._total_stake_held -= pool
        self._total_fees += fee

        if victor is None:
            self._agents[m.initiator].draws += 1
            self._agents[m.opponent].draws += 1
        else:
