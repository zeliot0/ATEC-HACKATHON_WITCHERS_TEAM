"""
Local Mission Advisor for Project C.R.A.T.E.R -- offline, no network calls.

Why local instead of a cloud LLM API:
  - Privacy/security: mission telemetry never leaves the rover's own
    compute. A deep-space or lunar-surface system has no reliable always-on
    uplink to a cloud API anyway -- the whole premise of this challenge is
    "no cloud, no GPS, no cellular."
  - Determinism: a hackathon demo needs an answer every single time, not
    "the API was rate-limited" or "the key expired" mid-pitch.
  - No external dependency, no API key to leak, no per-call cost.

What this actually is: a small retrieval-augmented system (RAG), not a
fine-tuned language model -- that distinction matters and we say so openly
rather than overclaiming. The "knowledge base" below is a short set of real
aerospace engineering principles (CCSDS telemetry framing, deep-space FEC
heritage, DSN link-budget practice, planetary-rover hazard-avoidance
doctrine, radiation effects on electronics) written in our own words from
well-known public domain aerospace engineering knowledge -- not copied from
any proprietary source. A lightweight TF-IDF retriever picks the most
relevant entry for the current link/pipeline state, and a small rule
engine turns live telemetry numbers into a concrete, numbers-backed
assessment. Swap this retriever for a local embedding model and the same
interface keeps working -- that's the natural "future work" upgrade path.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Knowledge base: short, factual aerospace engineering notes.
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    {
        "id": "ccsds_framing",
        "tags": "telemetry packet framing protocol overhead bandwidth",
        "text": (
            "CCSDS (Consultative Committee for Space Data Systems) telemetry "
            "framing is the real-world standard this project's binary "
            "FULL/DELTA packet split takes after: fixed-length frames, no "
            "length prefix needed, and the smallest possible header so "
            "every byte spent on overhead is a byte not spent on payload. "
            "On a bandwidth-starved link, packet overhead ratio matters as "
            "much as raw throughput."
        ),
    },
    {
        "id": "fec_heritage",
        "tags": "fec forward error correction parity loss recovery reed solomon voyager deep space",
        "text": (
            "Forward Error Correction has deep-space heritage going back to "
            "Voyager's Reed-Solomon coding: when a one-way light-time delay "
            "makes retransmission requests impractical, the receiver must "
            "be able to reconstruct lost data from redundancy sent "
            "alongside it. A single-parity XOR scheme (this project's "
            "choice) recovers exactly one erasure per coding block at very "
            "low overhead; a Reed-Solomon code trades more overhead for "
            "recovering multiple erasures per block -- the standard "
            "next step once XOR's one-loss-per-block ceiling is reached."
        ),
    },
    {
        "id": "dsn_link_budget",
        "tags": "link budget bandwidth latency loss degraded signal dsn deep space network",
        "text": (
            "Deep Space Network link budgets are planned around a worst-case "
            "signal margin, not an average one: latency, bandwidth, and "
            "packet loss are treated as a jointly degrading envelope, and "
            "operators plan for graceful degradation (lower-priority data "
            "dropped first) rather than a binary link-up/link-down model. "
            "A live loss-rate spike above roughly 50% is normally a sign "
            "either of a real outage or of a deliberately induced test "
            "condition, not of ordinary link noise."
        ),
    },
    {
        "id": "rover_hazard_doctrine",
        "tags": "hazard avoidance rover safety rock crater autonomy stop",
        "text": (
            "Planetary rover hazard-avoidance doctrine (as used on Mars "
            "rover missions) treats a newly detected, high-confidence "
            "nearby hazard as an immediate drive-stop condition rather "
            "than something to resolve later -- false positives cost time, "
            "but a missed hazard can end the mission. Hazard count and "
            "confidence trend are the two numbers operators watch most "
            "closely during an active traverse."
        ),
    },
    {
        "id": "radiation_effects",
        "tags": "solar flare radiation bit flip sensor noise electronics shielding",
        "text": (
            "Unshielded electronics in a high-radiation environment (a "
            "solar particle event, or prolonged exposure outside a "
            "planet's magnetosphere) see increased single-event upsets: "
            "bit flips in memory and transient sensor noise, exactly the "
            "kind of correlated salt-and-pepper corruption this project's "
            "flare-simulation models. Real missions respond with "
            "radiation-hardened hardware and redundant voting logic; here, "
            "the integrity check (CRC8) and FEC layer are what stand in "
            "for that defense at the software level."
        ),
    },
    {
        "id": "uncertainty_quantification",
        "tags": "uncertainty localization covariance position error confidence",
        "text": (
            "Standard practice in robotic localization is to report "
            "position estimates with an explicit uncertainty envelope "
            "rather than a bare point, because downstream planning "
            "decisions (how close to drive to a hazard) depend on how much "
            "to trust the estimate. An elongated uncertainty ellipse along "
            "the camera's line of sight versus across it reflects the real "
            "geometry of monocular ranging: depth is always less certain "
            "than lateral position."
        ),
    },
    {
        "id": "link_security",
        "tags": "encryption security integrity authentication tamper confidentiality key",
        "text": (
            "Mission telemetry links are typically secured with "
            "authenticated encryption so that a corrupted or tampered "
            "packet is rejected outright rather than silently accepted -- "
            "confidentiality and integrity are treated as one property "
            "(AEAD: Authenticated Encryption with Associated Data), not "
            "two separate mechanisms layered loosely together."
        ),
    },
]

_STOPWORDS = {"the", "a", "an", "is", "are", "to", "of", "and", "on", "in", "for", "this", "with", "as"}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS]


# TF-IDF index built once at import time over the small fixed KB -- no
# external ML library needed for a handful of documents.
_DOC_TOKENS = [_tokenize(e["tags"] + " " + e["text"]) for e in KNOWLEDGE_BASE]
_DOC_FREQ: Counter = Counter()
for _toks in _DOC_TOKENS:
    _DOC_FREQ.update(set(_toks))
_N_DOCS = len(KNOWLEDGE_BASE)


def _tfidf_vector(tokens: list[str]) -> Counter:
    tf = Counter(tokens)
    vec = Counter()
    for term, count in tf.items():
        df = _DOC_FREQ.get(term, 0)
        idf = math.log((_N_DOCS + 1) / (df + 1)) + 1.0
        vec[term] = count * idf
    return vec


_DOC_VECTORS = [_tfidf_vector(toks) for toks in _DOC_TOKENS]


def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve(query: str, top_k: int = 1) -> list[dict]:
    """Pure-Python TF-IDF retrieval over the local knowledge base."""
    qvec = _tfidf_vector(_tokenize(query))
    scored = [(_cosine(qvec, dv), entry) for dv, entry in zip(_DOC_VECTORS, KNOWLEDGE_BASE)]
    scored.sort(key=lambda p: p[0], reverse=True)
    return [entry for score, entry in scored[:top_k] if score > 0]


# ---------------------------------------------------------------------------
# Rule engine: turns live pipeline/link numbers into a concrete assessment,
# then grounds it with the most relevant retrieved knowledge-base entry.
# ---------------------------------------------------------------------------

def get_advice(stats: dict) -> str:
    """stats: a flat dict of current link + pipeline numbers (see server.py
    /api/ai_advice for the exact fields). Returns a 2-4 sentence, fully
    local, deterministic assessment -- no network call, ever."""
    loss_rate = float(stats.get("loss_rate", 0.0) or 0.0)
    latency_ms = float(stats.get("latency_ms", 0.0) or 0.0)
    survival_rate = float(stats.get("survival_rate", 1.0) or 1.0)
    fec_recoveries = int(stats.get("fec_recoveries", 0) or 0)
    fec_enabled = bool(stats.get("fec_enabled", True))
    unique_hazards = int(stats.get("unique_hazards", 0) or 0)
    total_packets = int(stats.get("total_packets_sent", 0) or 0)
    flare_active = bool(stats.get("flare_active", False))

    # Pick exactly ONE retrieval query from whichever single condition is
    # most salient right now (priority order below), so the grounding text
    # actually matches the headline instead of averaging several topics
    # together into whichever one happens to score highest on raw overlap.
    if flare_active:
        query = "radiation solar flare sensor noise bit flip electronics"
    elif loss_rate >= 0.3 and not fec_enabled:
        query = "link budget degraded severe loss bandwidth"
    elif fec_recoveries > 0:
        query = "fec forward error correction parity recovery reed solomon"
    elif loss_rate >= 0.3:
        query = "link budget degraded severe loss bandwidth"
    elif unique_hazards > 0:
        query = "hazard avoidance rover safety autonomy"
    else:
        query = "telemetry packet framing overhead protocol"
    hits = retrieve(query, top_k=1)
    grounding = hits[0]["text"] if hits else ""

    # Deterministic status line from the numbers themselves.
    if total_packets == 0:
        headline = "No telemetry observed yet -- mission has not started transmitting."
    elif flare_active:
        headline = (
            f"Solar flare event active: expect elevated sensor noise and bit-level "
            f"corruption. Current link survival is {survival_rate*100:.0f}%."
        )
    elif loss_rate >= 0.6:
        headline = (
            f"Link severely degraded ({loss_rate*100:.0f}% configured loss, "
            f"{latency_ms:.0f} ms latency). Survival is {survival_rate*100:.0f}%"
            + (f", with {fec_recoveries} packets reconstructed by FEC." if fec_enabled and fec_recoveries else ".")
        )
    elif loss_rate >= 0.3:
        headline = (
            f"Link degraded ({loss_rate*100:.0f}% loss). Survival holding at "
            f"{survival_rate*100:.0f}%"
            + (f" thanks to {fec_recoveries} FEC recoveries." if fec_enabled and fec_recoveries else " -- FEC is currently disabled, so losses are not being recovered.")
        )
    else:
        headline = (
            f"Link nominal ({loss_rate*100:.0f}% loss, {latency_ms:.0f} ms latency). "
            f"Survival at {survival_rate*100:.0f}%, {unique_hazards} hazards tracked."
        )

    recommendation = ""
    if not fec_enabled and loss_rate > 0.15:
        recommendation = " Recommendation: re-enable FEC -- at this loss rate it materially improves survival."
    elif flare_active:
        recommendation = " Recommendation: hold position until the flare clears; new detections are less reliable right now."
    elif loss_rate >= 0.6:
        recommendation = " Recommendation: prioritize high-confidence, near-range hazards only until the link recovers."

    return f"{headline}{recommendation} ({grounding})"
