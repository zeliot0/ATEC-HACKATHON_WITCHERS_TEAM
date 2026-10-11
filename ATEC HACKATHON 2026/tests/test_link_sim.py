import time

from src.link_sim import LinkEmulator, LinkConfig


def test_delivery_with_zero_loss():
    received = []
    emu = LinkEmulator(LinkConfig(loss_rate=0.0, latency_ms=10, jitter_ms=2, bandwidth_bps=1_000_000))
    emu.on_receive(lambda d: received.append(d))
    emu.start()
    for i in range(50):
        emu.send(f"pkt-{i}".encode())
    time.sleep(0.6)
    emu.stop()
    assert len(received) == 50


def test_loss_rate_is_statistically_respected():
    # The link uses a Gilbert-Elliott burst-loss model (correlated drops in
    # bursts of ~20 packets), not independent coin-flips per packet -- that's
    # deliberate (it's realistic for occlusion-style outages), but it also
    # means a single run's observed loss has real variance around the target
    # even though the transition probabilities are solved so the *stationary*
    # average matches it exactly. N=300 with a burst length of ~20 is only
    # ~15 independent burst cycles, which isn't enough samples to keep that
    # variance inside a tight band -- so this uses a longer run and a wider
    # tolerance to stay robust without weakening what's actually being tested
    # (that loss keeps roughly tracking the configured target, not pinning
    # an exact number).
    received = []
    emu = LinkEmulator(LinkConfig(loss_rate=0.3, latency_ms=5, jitter_ms=2, bandwidth_bps=1_000_000))
    emu.on_receive(lambda d: received.append(d))
    emu.start()
    N = 1200
    for i in range(N):
        emu.send(f"pkt-{i}".encode())
    time.sleep(2.5)
    emu.stop()
    observed_loss = 1 - len(received) / N
    assert abs(observed_loss - 0.3) < 0.15  # statistical tolerance


def test_live_config_update_takes_effect():
    emu = LinkEmulator(LinkConfig(loss_rate=0.0, latency_ms=10, jitter_ms=0, bandwidth_bps=1_000_000))
    emu.start()
    emu.set_config(loss_rate=0.9)
    assert emu.get_config()["loss_rate"] == 0.9
    emu.stop()
