import src.telemetry as T


def test_full_roundtrip():
    pkt = T.encode_full(hazard_id=42, seq=3, x_m=12.34, y_m=-5.67, sx_m=0.3, sy_m=0.8,
                         angle_deg=15.0, confidence=0.91, timestamp_ms=123456)
    assert len(pkt) == T.FULL_LEN
    d = T.decode_full(pkt)
    assert d["hazard_id"] == 42
    assert abs(d["x"] - 12.34) < 0.01
    assert abs(d["y"] + 5.67) < 0.01
    assert abs(d["confidence"] - 0.91) < 0.01


def test_delta_roundtrip():
    pkt = T.encode_delta(hazard_id=7, seq=1, dx_m=0.3, dy_m=-0.2, confidence=0.8, dt_ms=250)
    assert len(pkt) == T.DELTA_LEN
    d = T.decode_delta(pkt)
    assert abs(d["dx"] - 0.3) < 0.01
    assert abs(d["dy"] + 0.2) < 0.01


def test_crc_rejects_corruption():
    pkt = bytearray(T.encode_full(1, 0, 1.0, 1.0, 0.3, 0.3, 0.0, 0.5, 1000))
    pkt[-2] ^= 0xFF
    try:
        T.decode_full(bytes(pkt))
        assert False, "should have raised on corrupted CRC"
    except T.TelemetryError:
        pass


def test_fec_reconstructs_single_loss():
    builder = T.FECBlockBuilder(k=4)
    recon = T.FECReconstructor(k=4)
    pkts = [T.encode_full(i, i, float(i), float(i * 2), 0.3, 0.3, 0.0, 0.5, 1000 + i) for i in range(4)]
    parity = None
    for p in pkts:
        par = builder.add(p)
        if par is not None:
            parity = par
    assert parity is not None

    for i, p in enumerate(pkts):
        if i == 2:
            continue  # simulate dropping packet #2
        recon.observe_full(0, i, p)

    pd = T.decode_parity(parity)
    rebuilt = recon.observe_parity(0, pd["seq_mask"], pd["xor_payload"])
    assert rebuilt is not None
    decoded = T.decode_full(rebuilt)
    assert decoded["hazard_id"] == 2


def test_telemetry_encoder_switches_full_to_delta():
    enc = T.TelemetryEncoder()
    out1 = enc.encode(hazard_id=5, x_m=1.0, y_m=1.0, sx_m=0.3, sy_m=0.3, angle_deg=0.0,
                       confidence=0.6, distance_m=3.0, timestamp_ms=1000)
    assert out1[0][1][0] == T.MSG_FULL
    out2 = enc.encode(hazard_id=5, x_m=1.1, y_m=1.05, sx_m=0.3, sy_m=0.3, angle_deg=0.0,
                       confidence=0.6, distance_m=3.0, timestamp_ms=1200)
    assert out2[0][1][0] == T.MSG_DELTA


def test_compression_ratio_vs_json():
    full_pkt = T.encode_full(1, 0, 4.2, -1.1, 0.3, 0.3, 0.0, 0.87, 1000)
    json_bytes = T.JSON_BASELINE_EXAMPLE.format(hid=1, x=4.2, y=-1.1, c=0.87, t=1000).encode()
    assert len(full_pkt) < len(json_bytes) / 3
