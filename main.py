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
