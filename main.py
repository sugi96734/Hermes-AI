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
