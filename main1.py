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
            self._agents[victor].wins += 1
            self._agents[victor].tier_points += 12
            self._agents[victor].accrued_reward += fee // 2
            loser = m.opponent if victor == m.initiator else m.initiator
            self._agents[loser].losses += 1
            if self._agents[loser].tier_points >= 6:
                self._agents[loser].tier_points -= 6

    def get_agent(self, address: str) -> Optional[AgentRecord]:
        return self._agents.get(address)

    def get_match(self, match_id: int) -> Optional[MatchSlot]:
        return self._matches.get(match_id)

    def get_epoch(self, epoch_id: int) -> Optional[EpochMeta]:
        return self._epochs.get(epoch_id)

    def tier_for(self, address: str) -> int:
        a = self._agents.get(address)
        if not a:
            return 0
        tp = a.tier_points
        if tp >= 600:
            return 5
        if tp >= 240:
            return 4
        if tp >= 96:
            return 3
        if tp >= 24:
            return 2
        if tp >= 1:
            return 1
        return 0

    def claimable_reward(self, address: str) -> int:
        return self._agents.get(address, AgentRecord("", 0, 0, 0, 0, 0, 0, False)).accrued_reward

    def win_rate_bps(self, address: str) -> int:
        a = self._agents.get(address)
        if not a:
            return 0
        total = a.wins + a.losses + a.draws
        if total == 0:
            return 0
        return (a.wins * 10_000) // total

    def total_matches(self, address: str) -> int:
        a = self._agents.get(address)
        if not a:
            return 0
        return a.wins + a.losses + a.draws

    def build_leaderboard(self, epoch_id: int, max_size: int = 100) -> List[Tuple[str, int, int, int]]:
        out: List[Tuple[str, int, int, int]] = []
        for addr, a in self._agents.items():
            if not a.active:
                continue
            out.append((addr, a.tier_points, a.wins, a.losses))
        out.sort(key=lambda x: (-x[1], -x[2], x[3]))
        return out[:max_size]

    def encode_uint256(self, value: int) -> bytes:
        return value.to_bytes(32, "big")

    def encode_address(self, address: str) -> bytes:
        clean = address[2:] if address.startswith("0x") else address
        if len(clean) != 40:
            raise ValueError("Invalid address length")
        return bytes.fromhex(clean).rjust(32, b"\x00")

    def encode_bytes32(self, hex_str: str) -> bytes:
        clean = hex_str[2:] if hex_str.startswith("0x") else hex_str
        if len(clean) > 64:
            clean = clean[:64]
        else:
            clean = clean.zfill(64)
        return bytes.fromhex(clean)

    def selector_register_agent(self) -> str:
        return "0x" + hashlib.sha256(b"registerAgent(bytes32)").digest()[:4].hex()

    def selector_create_match(self) -> str:
        return "0x" + hashlib.sha256(b"createMatch(address,bytes32)").digest()[:4].hex()

    def selector_accept_match(self) -> str:
        return "0x" + hashlib.sha256(b"acceptMatch(uint256)").digest()[:4].hex()

    def selector_settle_match(self) -> str:
        return "0x" + hashlib.sha256(b"settleMatch(uint256,address,bytes32)").digest()[:4].hex()

    def selector_claim_reward(self) -> str:
        return "0x" + hashlib.sha256(b"claimReward()").digest()[:4].hex()

    def caduceus_proof_hash(self, match_id: int, nonce: str, chain_id: int, contract: str) -> str:
        payload = f"{CADUCEUS_NAMESPACE}{match_id}{nonce}{chain_id}{contract}"
        return "0x" + hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "genesis_block": self.genesis_block,
            "total_agents": self._total_agents,
            "total_matches": len(self._matches),
            "total_stake_held": self._total_stake_held,
            "total_fees": self._total_fees,
            "current_epoch_id": self._current_epoch_id,
        }


# ─── Tier names and config ───────────────────────────────────────────────────

TIER_NAMES = ["Unranked", "Bronze", "Silver", "Gold", "Platinum", "Caduceus"]


def tier_name(tier: int) -> str:
    if 0 <= tier < len(TIER_NAMES):
        return TIER_NAMES[tier]
    return "Unknown"


def format_wei(wei: int) -> str:
    s = str(wei)
    if len(s) <= 18:
        return "0." + s.zfill(18).rstrip("0") or "0"
    return str(wei // 10**18) + "." + s[-18:].rstrip("0").ljust(1, "0")


def parse_address(addr: str) -> str:
    if not addr:
        raise ValueError("Empty address")
    a = addr.strip()
    if a.startswith("0x"):
        a = a[2:]
    if len(a) != 40 or not all(c in "0123456789abcdefABCDEF" for c in a):
        raise ValueError("Invalid address")
    return "0x" + a.lower()


# ─── CLI and runner ───────────────────────────────────────────────────────────

def main() -> None:
    engine = HermesAIV2Engine(genesis_block=21_000_000)
    print("Hermes-AI V2 Engine")
    print("Version:", VERSION)
    print("Governor:", GOVERNOR)
    print("Namespace:", CADUCEUS_NAMESPACE[:20] + "...")

    engine.register_agent_local(GOVERNOR, "governor_bot")
    engine.register_agent_local(VAULT, "vault_bot")
    mid = engine.create_match_local(GOVERNOR, VAULT, FLOOR_STAKE_WEI)
    print("Created match id:", mid)
    engine.accept_match_local(mid, engine.genesis_block + 1)
    engine.settle_match_local(mid, GOVERNOR, engine.genesis_block + 2)
    print("Governor tier:", engine.tier_for(GOVERNOR))
    print("Governor claimable:", format_wei(engine.claimable_reward(GOVERNOR)))
    print("Leaderboard:", engine.build_leaderboard(1))
    print("Summary:", json.dumps(engine.to_dict(), indent=2))


# ─── Batch and pagination ─────────────────────────────────────────────────────

def get_match_ids_for_initiator(engine: HermesAIV2Engine, address: str, offset: int, limit: int) -> List[int]:
    ids = engine._matches_by_initiator.get(address, [])
    return ids[offset : offset + limit]


def get_match_ids_for_opponent(engine: HermesAIV2Engine, address: str, offset: int, limit: int) -> List[int]:
    ids = engine._matches_by_opponent.get(address, [])
    return ids[offset : offset + limit]


def get_open_match_ids(engine: HermesAIV2Engine, max_count: int) -> List[int]:
    out = []
    for mid, m in engine._matches.items():
        if m.state == MatchState.OPEN and len(out) < max_count:
            out.append(mid)
    return out


def get_settled_match_ids(engine: HermesAIV2Engine, offset: int, limit: int) -> List[int]:
    out = [mid for mid, m in engine._matches.items() if m.state == MatchState.SETTLED]
    return out[offset : offset + limit]


# ─── Config and constants export ─────────────────────────────────────────────

CONFIG = {
    "SCALE": SCALE,
    "FEE_BASIS": FEE_BASIS,
    "FEE_DENOM": FEE_DENOM,
    "MAX_PENDING": MAX_PENDING,
    "MATCH_TIMEOUT_BLOCKS": MATCH_TIMEOUT_BLOCKS,
    "EPOCH_BLOCKS": EPOCH_BLOCKS,
    "FLOOR_STAKE_WEI": FLOOR_STAKE_WEI,
    "REGISTER_FEE_WEI": REGISTER_FEE_WEI,
    "VERSION": VERSION,
    "GOVERNOR": GOVERNOR,
    "VAULT": VAULT,
    "ADJUDICATOR": ADJUDICATOR,
    "BOUNTY_POOL": BOUNTY_POOL,
    "RELAYER": RELAYER,
}


def get_config() -> Dict[str, Any]:
    return dict(CONFIG)


# ─── Event payload builders (for off-chain indexing) ──────────────────────────

def build_agent_registered_payload(address: str, name_hash: str, block: int) -> Dict[str, Any]:
    return {"event": "AgentRegistered", "agent": address, "nameHash": name_hash, "block": block}


def build_match_created_payload(match_id: int, initiator: str, opponent: str, stake: int, block: int) -> Dict[str, Any]:
    return {
        "event": "MatchCreated",
        "matchId": match_id,
        "initiator": initiator,
        "opponent": opponent,
        "stake": stake,
        "block": block,
    }


def build_match_settled_payload(match_id: int, victor: Optional[str], proof_hash: str, block: int) -> Dict[str, Any]:
    return {"event": "MatchSettled", "matchId": match_id, "victor": victor or "", "proofHash": proof_hash, "block": block}


# ─── Validation helpers ──────────────────────────────────────────────────────

def is_valid_eth_address(s: str) -> bool:
    if not s or len(s) < 42:
        return False
    clean = s[2:] if s.startswith("0x") else s
    return len(clean) == 40 and all(c in "0123456789abcdefABCDEF" for c in clean)


def is_valid_bytes32_hex(s: str) -> bool:
    if not s:
        return False
    clean = s[2:] if s.startswith("0x") else s
    return len(clean) <= 64 and all(c in "0123456789abcdefABCDEF" for c in clean)


# ─── Tier thresholds (match contract logic) ───────────────────────────────────

TIER_THRESHOLDS = [0, 1, 24, 96, 240, 600]


def tier_from_points(points: int) -> int:
    for i in range(len(TIER_THRESHOLDS) - 1, -1, -1):
        if points >= TIER_THRESHOLDS[i]:
            return i
    return 0


def min_points_for_tier(tier: int) -> int:
    if 0 <= tier < len(TIER_THRESHOLDS):
        return TIER_THRESHOLDS[tier]
    return 0


# ─── Fee and reward calculations ─────────────────────────────────────────────

def fee_from_pool(pool_wei: int) -> int:
    return (pool_wei * FEE_BASIS) // FEE_DENOM


def to_victor_amount(pool_wei: int) -> int:
    return pool_wei - fee_from_pool(pool_wei)


def accrued_reward_share(fee_wei: int) -> int:
    return fee_wei // 2


# ─── Epoch helpers ───────────────────────────────────────────────────────────

def epoch_end_block(epoch_id: int, genesis: int) -> int:
    return genesis + epoch_id * EPOCH_BLOCKS


def epoch_start_block(epoch_id: int, genesis: int) -> int:
    return genesis + (epoch_id - 1) * EPOCH_BLOCKS


def current_epoch_at_block(block: int, genesis: int) -> int:
    if block < genesis:
        return 0
    return 1 + (block - genesis) // EPOCH_BLOCKS


# ─── Match state labels ───────────────────────────────────────────────────────

MATCH_STATE_LABELS = {
    MatchState.OPEN: "open",
    MatchState.ACTIVE: "active",
    MatchState.SETTLED: "settled",
    MatchState.CANCELLED: "cancelled",
    MatchState.TIMED_OUT: "timed_out",
}


def match_state_label(state: MatchState) -> str:
    return MATCH_STATE_LABELS.get(state, "unknown")


# ─── Serialization for API responses ──────────────────────────────────────────

def agent_to_dict(agent: AgentRecord) -> Dict[str, Any]:
    return {
        "nameHash": agent.name_hash,
        "wins": agent.wins,
        "losses": agent.losses,
        "draws": agent.draws,
        "tierPoints": agent.tier_points,
        "registeredBlock": agent.registered_block,
        "accruedReward": agent.accrued_reward,
        "active": agent.active,
    }


def match_to_dict(m: MatchSlot) -> Dict[str, Any]:
    return {
        "matchId": m.match_id,
        "initiator": m.initiator,
        "opponent": m.opponent,
        "stakeWei": m.stake_wei,
        "createdBlock": m.created_block,
        "acceptedBlock": m.accepted_block,
        "state": match_state_label(m.state),
        "proofHash": m.proof_hash or "",
        "victor": m.victor or "",
    }


def epoch_to_dict(e: EpochMeta) -> Dict[str, Any]:
    return {
        "epochId": e.epoch_id,
        "startBlock": e.start_block,
        "endBlock": e.end_block,
        "matchCount": e.match_count,
        "boardRoot": e.board_root,
        "sealed": e.sealed,
    }


# ─── Extended engine methods ──────────────────────────────────────────────────

def get_agent_summary(engine: HermesAIV2Engine, address: str) -> Optional[Dict[str, Any]]:
    a = engine.get_agent(address)
    if not a:
        return None
    return {
        **agent_to_dict(a),
        "tier": engine.tier_for(address),
        "tierName": tier_name(engine.tier_for(address)),
        "winRateBps": engine.win_rate_bps(address),
        "totalMatches": engine.total_matches(address),
        "claimableReward": engine.claimable_reward(address),
    }


def get_match_summary(engine: HermesAIV2Engine, match_id: int) -> Optional[Dict[str, Any]]:
    m = engine.get_match(match_id)
    if not m:
        return None
    return match_to_dict(m)


# ─── Simulated RPC / contract call builders ───────────────────────────────────

def build_register_agent_calldata(name_hash_hex: str) -> str:
    if not name_hash_hex.startswith("0x"):
        name_hash_hex = "0x" + name_hash_hex
    return "0x" + name_hash_hex[2:].zfill(64)


def build_create_match_calldata(opponent: str, nonce_hash: str) -> Dict[str, str]:
    return {"opponent": opponent, "nonceHash": nonce_hash}


def build_accept_match_calldata(match_id: int) -> Dict[str, Any]:
    return {"matchId": match_id}


def build_settle_match_calldata(match_id: int, victor: str, proof_hash: str) -> Dict[str, Any]:
    return {"matchId": match_id, "victor": victor, "proofHash": proof_hash}


# ─── Bulk load and export ─────────────────────────────────────────────────────

def export_agents_csv(engine: HermesAIV2Engine) -> str:
    lines = ["address,nameHash,wins,losses,draws,tierPoints,accruedReward,active"]
    for addr, a in engine._agents.items():
        lines.append(f"{addr},{a.name_hash},{a.wins},{a.losses},{a.draws},{a.tier_points},{a.accrued_reward},{a.active}")
    return "\n".join(lines)


def export_matches_csv(engine: HermesAIV2Engine) -> str:
    lines = ["matchId,initiator,opponent,stakeWei,state,victor"]
    for mid, m in sorted(engine._matches.items()):
        lines.append(f"{mid},{m.initiator},{m.opponent},{m.stake_wei},{match_state_label(m.state)},{m.victor or ''}")
    return "\n".join(lines)


# ─── Stats aggregator ────────────────────────────────────────────────────────

def global_stats(engine: HermesAIV2Engine) -> Dict[str, Any]:
    open_count = sum(1 for m in engine._matches.values() if m.state == MatchState.OPEN)
    active_count = sum(1 for m in engine._matches.values() if m.state == MatchState.ACTIVE)
    settled_count = sum(1 for m in engine._matches.values() if m.state == MatchState.SETTLED)
    return {
        "totalAgents": engine._total_agents,
        "totalMatches": len(engine._matches),
        "openMatches": open_count,
        "activeMatches": active_count,
        "settledMatches": settled_count,
        "totalStakeHeld": engine._total_stake_held,
        "totalFees": engine._total_fees,
        "currentEpochId": engine._current_epoch_id,
    }


# ─── Reputation score (off-chain metric) ──────────────────────────────────────

def reputation_score(engine: HermesAIV2Engine, address: str) -> float:
    a = engine.get_agent(address)
    if not a:
        return 0.0
    total = a.wins + a.losses + a.draws
    if total == 0:
        return 0.0
    return (a.wins * 2.0 + a.draws) / (total * 2.0)


# ─── Timeout check ───────────────────────────────────────────────────────────

def is_match_timed_out(m: MatchSlot, current_block: int) -> bool:
    return m.state == MatchState.ACTIVE and current_block >= m.accepted_block + MATCH_TIMEOUT_BLOCKS


def blocks_until_timeout(m: MatchSlot, current_block: int) -> int:
    if m.state != MatchState.ACTIVE:
        return 0
    deadline = m.accepted_block + MATCH_TIMEOUT_BLOCKS
    if current_block >= deadline:
        return 0
    return deadline - current_block


# ─── Additional engine: replay from events ─────────────────────────────────────

class HermesAIV2Replay:
    """Replay engine state from a list of event-like dicts."""

    def __init__(self, genesis_block: int = 0):
        self.engine = HermesAIV2Engine(genesis_block)
        self._block = genesis_block

    def advance_block(self, n: int = 1) -> None:
        self._block += n

    def apply_register(self, address: str, name: str) -> None:
        self.engine.register_agent_local(address, name)

    def apply_create_match(self, initiator: str, opponent: str, stake_wei: int) -> int:
        return self.engine.create_match_local(initiator, opponent, stake_wei)

    def apply_accept_match(self, match_id: int) -> None:
        self.engine.accept_match_local(match_id, self._block)
        self.advance_block()

    def apply_settle_match(self, match_id: int, victor: Optional[str]) -> None:
        self.engine.settle_match_local(match_id, victor, self._block)
        self.advance_block()


# ─── Main extended ───────────────────────────────────────────────────────────

def run_demo() -> None:
    engine = HermesAIV2Engine(genesis_block=21_000_000)
    engine.register_agent_local(GOVERNOR, "alpha")
    engine.register_agent_local(VAULT, "beta")
    engine.register_agent_local(ADJUDICATOR, "gamma")

    m1 = engine.create_match_local(GOVERNOR, VAULT, FLOOR_STAKE_WEI)
    engine.accept_match_local(m1, 21_000_001)
    engine.settle_match_local(m1, GOVERNOR, 21_000_002)

    m2 = engine.create_match_local(VAULT, ADJUDICATOR, FLOOR_STAKE_WEI)
    engine.accept_match_local(m2, 21_000_003)
    engine.settle_match_local(m2, None, 21_000_004)

    print("Global stats:", global_stats(engine))
    print("Governor summary:", get_agent_summary(engine, GOVERNOR))
    print("Leaderboard:", engine.build_leaderboard(1))
    print("Reputation GOVERNOR:", reputation_score(engine, GOVERNOR))


# ─── ABI encoding helpers (32-byte words) ─────────────────────────────────────

def abi_encode_uint256(value: int) -> bytes:
    """Encode uint256 as 32 bytes big-endian."""
    return value.to_bytes(32, "big")


def abi_encode_int256(value: int) -> bytes:
    """Encode int256 as 32 bytes two's complement."""
    return value.to_bytes(32, "big", signed=True)


def abi_encode_address(addr: str) -> bytes:
    """Encode address as 20 bytes right-padded to 32."""
    clean = addr[2:] if addr.startswith("0x") else addr
    return bytes.fromhex(clean).rjust(32, b'\x00')


def abi_encode_bytes32(hex_val: str) -> bytes:
    """Encode bytes32 from hex string."""
    clean = hex_val[2:] if hex_val.startswith("0x") else hex_val
    return bytes.fromhex(clean.zfill(64)[:64])


def abi_encode_bool(value: bool) -> bytes:
    """Encode bool as 32 bytes (0 or 1 in last byte)."""
    return (1 if value else 0).to_bytes(32, "big")


# ─── Selector computation (first 4 bytes of keccak256) ────────────────────────

def keccak256_hex(data: bytes) -> str:
    """Return keccak256 hash as hex (SHA-256 used as stand-in)."""
    return "0x" + hashlib.sha256(data).hexdigest()


def function_selector(signature: str) -> str:
    """Return 4-byte selector for a function signature."""
    h = hashlib.sha256(signature.encode()).digest()
    return "0x" + h[:4].hex()


SELECTORS = {
    "registerAgent": function_selector("registerAgent(bytes32)"),
    "createMatch": function_selector("createMatch(address,bytes32)"),
    "acceptMatch": function_selector("acceptMatch(uint256)"),
    "settleMatch": function_selector("settleMatch(uint256,address,bytes32)"),
    "claimReward": function_selector("claimReward()"),
    "cancelMatch": function_selector("cancelMatch(uint256)"),
    "claimTimeout": function_selector("claimTimeout(uint256)"),
    "startNextEpoch": function_selector("startNextEpoch()"),
    "sealEpoch": function_selector("sealEpoch(uint256,bytes32)"),
    "recordEpochBoard": function_selector("recordEpochBoard(uint256,address[])"),
    "setFrozen": function_selector("setFrozen(bool)"),
}


# ─── Pagination helpers ──────────────────────────────────────────────────────

def paginate(items: List[Any], page: int, per_page: int) -> Tuple[List[Any], int]:
    """Return (slice of items for page, total_pages)."""
    total = len(items)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end], total_pages


# ─── Leaderboard with rank ────────────────────────────────────────────────────

def leaderboard_with_rank(engine: HermesAIV2Engine, epoch_id: int, max_size: int = 100) -> List[Dict[str, Any]]:
    """Return leaderboard entries with 1-based rank."""
    raw = engine.build_leaderboard(epoch_id, max_size)
    return [
        {"rank": i + 1, "address": addr, "tierPoints": tp, "wins": w, "losses": l, "tier": engine.tier_for(addr)}
        for i, (addr, tp, w, l) in enumerate(raw)
    ]


# ─── Config validation ───────────────────────────────────────────────────────

def validate_register_params(address: str, name: str, value_wei: int) -> List[str]:
    errors = []
    if not is_valid_eth_address(address):
        errors.append("Invalid agent address")
    if not name or len(name) > 64:
        errors.append("Invalid name length")
    if value_wei < REGISTER_FEE_WEI:
        errors.append("Insufficient registration fee")
    return errors


def validate_create_match_params(initiator: str, opponent: str, stake_wei: int, engine: HermesAIV2Engine) -> List[str]:
    errors = []
    if not is_valid_eth_address(initiator) or not is_valid_eth_address(opponent):
        errors.append("Invalid address")
    if initiator == opponent:
        errors.append("Cannot match self")
    if stake_wei < FLOOR_STAKE_WEI:
        errors.append("Stake below minimum")
    if initiator not in engine._agents:
        errors.append("Initiator not registered")
    if opponent not in engine._agents:
        errors.append("Opponent not registered")
    pending = len([m for m in engine._matches_by_initiator.get(initiator, []) if engine._matches[m].state == MatchState.OPEN])
    if pending >= MAX_PENDING:
        errors.append("Pending match limit reached")
    return errors


# ─── Snapshot and diff (for debugging / audit) ────────────────────────────────

def snapshot_agents(engine: HermesAIV2Engine) -> Dict[str, Dict[str, Any]]:
    """Return full snapshot of all agents as dict."""
    return {addr: agent_to_dict(a) for addr, a in engine._agents.items()}


def snapshot_matches(engine: HermesAIV2Engine) -> Dict[int, Dict[str, Any]]:
    """Return full snapshot of all matches as dict."""
    return {mid: match_to_dict(m) for mid, m in engine._matches.items()}


# ─── Wei / Ether conversion ──────────────────────────────────────────────────

def wei_to_ether(wei: int) -> float:
    return wei / 10**18


def ether_to_wei(ether: float) -> int:
    return int(ether * 10**18)


# ─── Block time estimates (12s per block assumption) ──────────────────────────

SECONDS_PER_BLOCK = 12


def blocks_to_seconds(blocks: int) -> int:
    return blocks * SECONDS_PER_BLOCK


def blocks_to_days(blocks: int) -> float:
    return blocks * SECONDS_PER_BLOCK / 86400


def epoch_duration_days() -> float:
    return blocks_to_days(EPOCH_BLOCKS)


def match_timeout_seconds() -> int:
    return blocks_to_seconds(MATCH_TIMEOUT_BLOCKS)


# ─── Event log parsers (simulated) ────────────────────────────────────────────

def parse_agent_registered(log: Dict[str, Any]) -> Optional[Tuple[str, str, int]]:
    """Return (agent, nameHash, block) if valid."""
    if log.get("event") != "AgentRegistered":
        return None
    a = log.get("agent")
    nh = log.get("nameHash")
    b = log.get("block")
    if a and nh is not None and b is not None:
        return (a, nh, int(b))
    return None


def parse_match_created(log: Dict[str, Any]) -> Optional[Tuple[int, str, str, int, int]]:
    """Return (matchId, initiator, opponent, stake, block) if valid."""
    if log.get("event") != "MatchCreated":
        return None
    mid = log.get("matchId")
    init = log.get("initiator")
    opp = log.get("opponent")
    stake = log.get("stake")
    b = log.get("block")
    if mid is not None and init and opp and stake is not None and b is not None:
        return (int(mid), init, opp, int(stake), int(b))
    return None


# ─── Health check (engine invariants) ──────────────────────────────────────────

def check_invariants(engine: HermesAIV2Engine) -> List[str]:
    """Check basic invariants; return list of violation messages."""
    issues = []
    total_stake = sum(
        m.stake_wei * (2 if m.state == MatchState.ACTIVE else 1)
        for m in engine._matches.values()
        if m.state in (MatchState.OPEN, MatchState.ACTIVE)
    )
    if total_stake != engine._total_stake_held:
        issues.append("total_stake_held mismatch")
    if engine._total_agents != len(engine._agents):
        issues.append("total_agents count mismatch")
    return issues


# ─── Default addresses (for tests / scripts) ──────────────────────────────────

DEFAULT_GENESIS = 21_000_000


def create_engine_with_fixtures(genesis: int = DEFAULT_GENESIS) -> HermesAIV2Engine:
    """Create engine and register governor, vault, adjudicator."""
    e = HermesAIV2Engine(genesis)
    e.register_agent_local(GOVERNOR, "governor")
    e.register_agent_local(VAULT, "vault")
    e.register_agent_local(ADJUDICATOR, "adjudicator")
    return e


# ─── JSON export/import stubs ─────────────────────────────────────────────────

def engine_state_to_json(engine: HermesAIV2Engine) -> str:
    """Export engine state as JSON string (agents + matches + epochs)."""
    data = {
        "version": VERSION,
        "genesis_block": engine.genesis_block,
        "next_match_id": engine._next_match_id,
        "current_epoch_id": engine._current_epoch_id,
        "total_stake_held": engine._total_stake_held,
        "total_fees": engine._total_fees,
        "total_agents": engine._total_agents,
        "agents": {addr: agent_to_dict(a) for addr, a in engine._agents.items()},
        "matches": {str(mid): match_to_dict(m) for mid, m in engine._matches.items()},
        "epochs": {str(eid): epoch_to_dict(e) for eid, e in engine._epochs.items()},
    }
    return json.dumps(data, indent=2)


# ─── Entry points for external tools ───────────────────────────────────────────

def get_all_selectors() -> Dict[str, str]:
    return dict(SELECTORS)


def get_tier_info() -> List[Dict[str, Any]]:
    return [
        {"tier": i, "name": tier_name(i), "minPoints": min_points_for_tier(i)}
        for i in range(len(TIER_NAMES))
    ]


# ─── Batch agent summaries ───────────────────────────────────────────────────

def get_agent_summaries_batch(engine: HermesAIV2Engine, addresses: List[str]) -> List[Optional[Dict[str, Any]]]:
    return [get_agent_summary(engine, addr) for addr in addresses]


def get_match_summaries_batch(engine: HermesAIV2Engine, match_ids: List[int]) -> List[Optional[Dict[str, Any]]]:
    return [get_match_summary(engine, mid) for mid in match_ids]


# ─── Caduceus proof verification (off-chain) ──────────────────────────────────

def verify_caduceus_proof(engine: HermesAIV2Engine, match_id: int, nonce: str, chain_id: int, contract: str, expected_hash: str) -> bool:
    computed = engine.caduceus_proof_hash(match_id, nonce, chain_id, contract)
    return computed.lower() == expected_hash.lower()


# ─── Reward projection ───────────────────────────────────────────────────────

def project_reward_after_win(engine: HermesAIV2Engine, address: str, pool_wei: int) -> int:
    """Project accrued reward share after winning a match with given pool size."""
    fee = fee_from_pool(pool_wei)
    return engine.claimable_reward(address) + accrued_reward_share(fee)


# ─── Match history for agent ──────────────────────────────────────────────────

def get_agent_match_history(engine: HermesAIV2Engine, address: str, limit: int = 50) -> List[Dict[str, Any]]:
    init_ids = engine._matches_by_initiator.get(address, [])
    opp_ids = engine._matches_by_opponent.get(address, [])
    all_ids = sorted(set(init_ids) | set(opp_ids), reverse=True)[:limit]
    return [match_to_dict(engine._matches[mid]) for mid in all_ids if mid in engine._matches]


# ─── Epoch board builder (for sealing) ────────────────────────────────────────

def build_epoch_board_root(agent_addresses: List[str]) -> str:
    """Build a simple root hash of ordered addresses (stand-in for Merkle)."""
    payload = "".join(a.lower() for a in agent_addresses)
    return "0x" + hashlib.sha256(payload.encode()).hexdigest()


# ─── Stats per epoch ─────────────────────────────────────────────────────────

def get_epoch_stats(engine: HermesAIV2Engine, epoch_id: int) -> Dict[str, Any]:
    e = engine.get_epoch(epoch_id)
    if not e:
        return {}
    matches_in_epoch = [m for m in engine._matches.values() if e.start_block <= m.created_block < e.end_block]
    return {
        "epochId": epoch_id,
        "startBlock": e.start_block,
        "endBlock": e.end_block,
        "matchCount": e.match_count,
        "sealed": e.sealed,
        "boardRoot": e.board_root,
        "matchesSettled": sum(1 for m in matches_in_epoch if m.state == MatchState.SETTLED),
    }


# ─── CLI argument helpers ─────────────────────────────────────────────────────

def parse_block_arg(s: str) -> int:
    return int(s, 0)


def parse_wei_arg(s: str) -> int:
    if s.endswith("ether") or s.endswith("eth"):
        return ether_to_wei(float(s.replace("ether", "").replace("eth", "").strip()))
    return int(s, 0)


# ─── Hex and bytes utilities ──────────────────────────────────────────────────

def to_hex(b: bytes) -> str:
    return "0x" + b.hex()


def from_hex(s: str) -> bytes:
    clean = s[2:] if s.startswith("0x") else s
    return bytes.fromhex(clean)


# ─── Time-based estimates ─────────────────────────────────────────────────────

def estimate_blocks_until_epoch_end(engine: HermesAIV2Engine, current_block: int) -> int:
    e = engine.get_epoch(engine._current_epoch_id)
    if not e or current_block >= e.end_block:
        return 0
    return e.end_block - current_block


# ─── Sanity checks for contract alignment ──────────────────────────────────────

def assert_contract_constants() -> None:
    """Assert that Python constants match documented contract values."""
    assert FEE_BASIS == 920
    assert FEE_DENOM == 10_000
    assert MAX_PENDING == 48
    assert MATCH_TIMEOUT_BLOCKS == 9600
    assert EPOCH_BLOCKS == 302400
    assert VERSION == 2


# ─── Demo with multiple matches ───────────────────────────────────────────────

def run_stress_demo(n_matches: int = 20) -> None:
    engine = HermesAIV2Engine(genesis_block=100_000)
    addrs = [GOVERNOR, VAULT, ADJUDICATOR]
