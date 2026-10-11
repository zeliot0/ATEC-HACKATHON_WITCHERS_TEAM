"""
Telemetry protocol for Project C.R.A.T.E.R.

Design rationale (for the jury / Q&A):
- Fixed-point integers instead of floats: cheaper to pack, deterministic size,
  and matches how real spacecraft telemetry formats (e.g. CCSDS packets) work.
- Two packet kinds: FULL (complete state) and DELTA (cheap incremental update).
  Re-observing the same hazard with a small position change costs 9 bytes
  instead of 14 -- on a bandwidth-starved deep-space-style link that matters.
- Forward Error Correction (XOR parity over blocks of FULL packets): on a
  one-way or high-latency link, asking for a retransmit may not be possible
  in time. A single parity packet lets the receiver reconstruct exactly one
  lost packet per block of K, with zero retransmission round trip.
- CRC8 on every packet: corrupted packets must never silently masquerade as
  a valid, wrong position -- CRC is a last line of integrity defense before
  anything reaches Mission Control.

All packets are fixed-length per type, so no length prefix is needed and
parsing is branch-free once the first byte (msg_type) is read.
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MSG_FULL = 0x01
MSG_DELTA = 0x02
MSG_PARITY = 0x03

POS_SCALE = 100.0          # fixed-point scale: 1 unit = 1 cm (int16 -> +-327.67 m)
DELTA_SCALE = 100.0        # delta also in cm, but stored as int8 (+-1.27 m per update)
CONF_SCALE = 255.0         # confidence 0..1 -> 0..255

FULL_FMT = ">BBHhhBBbBI"      # msg_type, seq, hazard_id, x_cm, y_cm, sx_dm, sy_dm, angle_deg, confidence, timestamp_ms
FULL_LEN = struct.calcsize(FULL_FMT) + 1   # +1 for trailing CRC8 byte

DELTA_FMT = ">BBHbbBB"     # msg_type, seq, hazard_id, dx_cm, dy_cm, confidence, dt_100ms
DELTA_LEN = struct.calcsize(DELTA_FMT) + 1

FEC_K = 4  # one parity packet protects every 4 FULL packets
PARITY_FMT = ">BHB"        # msg_type, block_id, seq_mask
PARITY_LEN = struct.calcsize(PARITY_FMT) + (FULL_LEN - 1) + 1  # + xor payload + crc

JSON_BASELINE_EXAMPLE = (
    '{{"hazard_id": {hid}, "x": {x}, "y": {y}, "confidence": {c}, "timestamp": {t}}}'
)


# ---------------------------------------------------------------------------
# CRC8 (polynomial 0x07, no reflection) -- small, fast, table-driven
# ---------------------------------------------------------------------------

def _build_crc8_table(poly: int = 0x07):
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
        table.append(crc)
    return table


_CRC8_TABLE = _build_crc8_table()


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _CRC8_TABLE[(crc ^ b) & 0xFF]
    return crc


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

class TelemetryError(ValueError):
    pass


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def encode_full(hazard_id: int, seq: int, x_m: float, y_m: float, sx_m: float, sy_m: float, angle_deg: float,
                 confidence: float, timestamp_ms: int) -> bytes:
    x_cm = int(clamp(round(x_m * POS_SCALE), -32768, 32767))
    y_cm = int(clamp(round(y_m * POS_SCALE), -32768, 32767))
    sx_dm = int(clamp(round(sx_m * 10.0), 0, 255))
    sy_dm = int(clamp(round(sy_m * 10.0), 0, 255))
    a_deg = int(clamp(round(angle_deg), -128, 127))
    conf_b = int(clamp(round(confidence * CONF_SCALE), 0, 255))
    ts = int(timestamp_ms) & 0xFFFFFFFF
    body = struct.pack(FULL_FMT, MSG_FULL, seq & 0xFF, hazard_id & 0xFFFF,
                        x_cm, y_cm, sx_dm, sy_dm, a_deg, conf_b, ts)
    return body + bytes([crc8(body)])


def decode_full(packet: bytes) -> dict:
    if len(packet) != FULL_LEN:
        raise TelemetryError(f"bad FULL length {len(packet)}")
    body, crc_byte = packet[:-1], packet[-1]
    if crc8(body) != crc_byte:
        raise TelemetryError("CRC mismatch on FULL packet")
    msg_type, seq, hazard_id, x_cm, y_cm, sx_dm, sy_dm, a_deg, conf_b, ts = struct.unpack(FULL_FMT, body)
    if msg_type != MSG_FULL:
        raise TelemetryError("not a FULL packet")
    return {
        "type": "FULL", "seq": seq, "hazard_id": hazard_id,
        "x": x_cm / POS_SCALE, "y": y_cm / POS_SCALE,
        "sx": sx_dm / 10.0, "sy": sy_dm / 10.0, "angle": float(a_deg),
        "confidence": conf_b / CONF_SCALE, "timestamp_ms": ts,
    }


def encode_delta(hazard_id: int, seq: int, dx_m: float, dy_m: float,
                  confidence: float, dt_ms: int) -> bytes:
    dx_cm = int(clamp(round(dx_m * DELTA_SCALE), -128, 127))
    dy_cm = int(clamp(round(dy_m * DELTA_SCALE), -128, 127))
    conf_b = int(clamp(round(confidence * CONF_SCALE), 0, 255))
    dt_units = int(clamp(round(dt_ms / 100.0), 0, 255))  # 100ms resolution, up to 25.5s
    body = struct.pack(DELTA_FMT, MSG_DELTA, seq & 0xFF, hazard_id & 0xFFFF,
                        dx_cm, dy_cm, conf_b, dt_units)
    return body + bytes([crc8(body)])


def decode_delta(packet: bytes) -> dict:
    if len(packet) != DELTA_LEN:
        raise TelemetryError(f"bad DELTA length {len(packet)}")
    body, crc_byte = packet[:-1], packet[-1]
    if crc8(body) != crc_byte:
        raise TelemetryError("CRC mismatch on DELTA packet")
    msg_type, seq, hazard_id, dx_cm, dy_cm, conf_b, dt_units = struct.unpack(DELTA_FMT, body)
    if msg_type != MSG_DELTA:
        raise TelemetryError("not a DELTA packet")
    return {
        "type": "DELTA", "seq": seq, "hazard_id": hazard_id,
        "dx": dx_cm / DELTA_SCALE, "dy": dy_cm / DELTA_SCALE,
        "confidence": conf_b / CONF_SCALE, "dt_ms": dt_units * 100,
    }


def encode_parity(block_id: int, seq_mask: int, xor_payload: bytes) -> bytes:
    assert len(xor_payload) == FULL_LEN - 1, "parity payload must match a FULL body length"
    body = struct.pack(PARITY_FMT, MSG_PARITY, block_id & 0xFFFF, seq_mask & 0xFF) + xor_payload
    return body + bytes([crc8(body)])


def decode_parity(packet: bytes) -> dict:
    if len(packet) != PARITY_LEN:
        raise TelemetryError(f"bad PARITY length {len(packet)}")
    body, crc_byte = packet[:-1], packet[-1]
    if crc8(body) != crc_byte:
        raise TelemetryError("CRC mismatch on PARITY packet")
    header_len = struct.calcsize(PARITY_FMT)
    msg_type, block_id, seq_mask = struct.unpack(PARITY_FMT, body[:header_len])
    if msg_type != MSG_PARITY:
        raise TelemetryError("not a PARITY packet")
    xor_payload = body[header_len:]
    return {"type": "PARITY", "block_id": block_id, "seq_mask": seq_mask, "xor_payload": xor_payload}


def peek_type(packet: bytes) -> Optional[int]:
    if not packet:
        return None
    return packet[0]


def decode_any(packet: bytes) -> dict:
    t = peek_type(packet)
    if t == MSG_FULL:
        return decode_full(packet)
    if t == MSG_DELTA:
        return decode_delta(packet)
    if t == MSG_PARITY:
        return decode_parity(packet)
    raise TelemetryError(f"unknown msg_type {t}")


# ---------------------------------------------------------------------------
# FEC block builder (sender side) / reconstructor (receiver side)
# ---------------------------------------------------------------------------

class FECBlockBuilder:
    """Accumulates FULL packets into blocks of FEC_K and emits one XOR parity
    packet per block. Only FULL packets are FEC-protected -- they carry the
    authoritative position and are worth the extra 18 bytes; DELTA packets
    are cheap refinements and are left unprotected by design."""

    def __init__(self, k: int = FEC_K):
        self.k = k
        self.block_id = 0
        self._bodies: list[bytes] = []  # raw FULL packets (with CRC) in current block
        self._mask = 0

    def add(self, full_packet: bytes) -> Optional[bytes]:
        idx = len(self._bodies)
        self._bodies.append(full_packet)
        self._mask |= (1 << idx)
        if len(self._bodies) == self.k:
            parity_payload = bytearray(FULL_LEN - 1)
            for pkt in self._bodies:
                body = pkt[:-1]  # exclude CRC byte from XOR so length matches FULL_LEN-1
                for i, b in enumerate(body):
                    parity_payload[i] ^= b
            parity_pkt = encode_parity(self.block_id, self._mask, bytes(parity_payload))
            self.block_id = (self.block_id + 1) & 0xFFFF
            self._bodies = []
            self._mask = 0
            return parity_pkt
        return None


class FECReconstructor:
    """Receiver-side companion: buffers packets per block_id; if exactly one
    slot is missing when the parity packet arrives, reconstructs it."""

    def __init__(self, k: int = FEC_K):
        self.k = k
        self._blocks: dict[int, dict[int, bytes]] = {}

    def observe_full(self, block_id: int, seq_in_block: int, raw_packet: bytes):
        self._blocks.setdefault(block_id, {})[seq_in_block] = raw_packet

    def observe_parity(self, block_id: int, seq_mask: int, xor_payload: bytes) -> Optional[bytes]:
        got = self._blocks.get(block_id, {})
        missing = [i for i in range(self.k) if (seq_mask >> i) & 1 and i not in got]
        if len(missing) != 1:
            self._blocks.pop(block_id, None)
            return None
        acc = bytearray(xor_payload)
        for i, pkt in got.items():
            body = pkt[:-1]
            for j, b in enumerate(body):
                acc[j] ^= b
        self._blocks.pop(block_id, None)
        body = bytes(acc)
        return body + bytes([crc8(body)])


# ---------------------------------------------------------------------------
# Priority + delta-tracking encoder (what the pipeline actually calls)
# ---------------------------------------------------------------------------

@dataclass
class _HazardState:
    last_x: float
    last_y: float
    last_full_ts: int
    seq: int = 0


class TelemetryEncoder:
    """Stateful encoder used by the pipeline: decides FULL vs DELTA, assigns
    FEC blocks, and scores packets for priority ordering."""

    FULL_REFRESH_MS = 5000      # force a FULL refresh at least this often
    MAX_DELTA_M = 1.2           # beyond this movement, DELTA can't represent it -> FULL

    def __init__(self):
        self._states: dict[int, _HazardState] = {}
        self._fec = FECBlockBuilder()
        self._seq_counter = 0

    def encode(self, hazard_id: int, x_m: float, y_m: float, sx_m: float, sy_m: float, angle_deg: float,
               confidence: float, distance_m: float, timestamp_ms: int):
        """Returns a list of (priority, packet_bytes, redundant: bool)."""
        out = []
        st = self._states.get(hazard_id)
        seq = self._seq_counter & 0xFF
        self._seq_counter += 1

        needs_full = (
            st is None
            or (timestamp_ms - st.last_full_ts) > self.FULL_REFRESH_MS
            or abs(x_m - st.last_x) > self.MAX_DELTA_M
            or abs(y_m - st.last_y) > self.MAX_DELTA_M
        )

        priority = confidence * 0.6 + (1.0 / (1.0 + max(distance_m, 0.0))) * 0.4
        redundant = confidence >= 0.8 and distance_m <= 5.0

        if needs_full:
            pkt = encode_full(hazard_id, seq, x_m, y_m, sx_m, sy_m, angle_deg, confidence, timestamp_ms)
            self._states[hazard_id] = _HazardState(x_m, y_m, timestamp_ms, seq)
            out.append((priority, pkt, redundant))
            parity = self._fec.add(pkt)
            if parity is not None:
                out.append((1.0, parity, False))  # parity always sent, max priority
        else:
            dx, dy = x_m - st.last_x, y_m - st.last_y
            dt = timestamp_ms - st.last_full_ts
            pkt = encode_delta(hazard_id, seq, dx, dy, confidence, dt)
            out.append((priority, pkt, redundant))
        return out
