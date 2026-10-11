#!/usr/bin/env python3
"""
Produces real, checkable numbers for every graded category in the
preselection rubric. Run after setup (`pip install -r requirements.txt`)
with `python benchmark.py`. Writes benchmark_results.json and prints a
summary table -- this is the evidence a jury reading the repo can verify
without running the live demo.
"""
from __future__ import annotations

import json
import statistics
import time

import numpy as np

from src import telemetry as T
from src.link_sim import LinkEmulator, LinkConfig
from src.localization import CameraModel, RoverPose, pixel_to_landing_frame
from src.perception import generate_synthetic_frame, detect_hazards

IMAGE_W, IMAGE_H = 480, 360
RESULTS: dict = {}


def bench_perception(n_frames=120):
    frames_gt = [generate_synthetic_frame(width=IMAGE_W, height=IMAGE_H, n_hazards=6, seed=i)
                 for i in range(n_frames)]

    t0 = time.perf_counter()
    all_detections = []
    for frame, _gt in frames_gt:
        all_detections.append(detect_hazards(frame))
    t1 = time.perf_counter()
    fps = n_frames / (t1 - t0)

    # precision/recall: match detections to ground truth within 2x the GT radius
    tp = fp = fn = 0
    for (_frame, gt), dets in zip(frames_gt, all_detections):
        matched_gt = set()
        for d in dets:
            hit = False
            for i, g in enumerate(gt):
                if i in matched_gt:
                    continue
                dist = ((d.px - g["px"]) ** 2 + (d.py - g["py"]) ** 2) ** 0.5
                if dist < max(2 * g["radius_px"], 10):
                    matched_gt.add(i)
                    hit = True
                    break
            if hit:
                tp += 1
            else:
                fp += 1
        fn += len(gt) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    RESULTS["perception"] = {
        "fps": round(fps, 1), "frames_tested": n_frames,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
    }


def bench_telemetry():
    full_pkt = T.encode_full(hazard_id=12, seq=0, x_m=4.2, y_m=-1.1, confidence=0.87, timestamp_ms=1000)
    delta_pkt = T.encode_delta(hazard_id=12, seq=1, dx_m=0.2, dy_m=-0.1, confidence=0.85, dt_ms=400)
    json_equiv = T.JSON_BASELINE_EXAMPLE.format(hid=12, x=4.2, y=-1.1, c=0.87, t=1000).encode()

    RESULTS["telemetry"] = {
        "full_packet_bytes": len(full_pkt),
        "delta_packet_bytes": len(delta_pkt),
        "json_equivalent_bytes": len(json_equiv),
        "compression_ratio_full_vs_json": round(len(json_equiv) / len(full_pkt), 2),
        "compression_ratio_delta_vs_json": round(len(json_equiv) / len(delta_pkt), 2),
        "fec_overhead_bytes_per_block": T.PARITY_LEN,
        "fec_block_size": T.FEC_K,
    }


def bench_fec_and_link(n_packets=400, trials_loss=(0.0, 0.1, 0.3, 0.5, 0.7)):
    """Measures three distinct things, kept separate on purpose:
      - link_level_packet_survival: ALL packets sent over the wire (FULL +
        PARITY), straight from the emulator's own counters -- the ground
        truth for "did the socket layer deliver the byte".
      - raw_data_survival_rate: of just the FULL hazard packets, how many
        arrived with no FEC help at all.
      - survival_rate_with_fec: of the FULL hazard packets, how many are
        either directly received OR recoverable from a surviving parity
        packet -- this is the number that matters operationally.
    """
    results = []
    for loss in trials_loss:
        received_list: list[bytes] = []
        emu = LinkEmulator(LinkConfig(loss_rate=loss, latency_ms=20, jitter_ms=5, bandwidth_bps=1_000_000))
        emu.on_receive(lambda d: received_list.append(d))
        emu.start()

        builder = T.FECBlockBuilder(k=T.FEC_K)
        sent_full = []
        parity_by_block: dict[int, bytes] = {}
        for i in range(n_packets):
            pkt = T.encode_full(hazard_id=i % 50, seq=i, x_m=float(i % 20), y_m=float(i % 15),
                                 confidence=0.7, timestamp_ms=1000 + i)
            sent_full.append(pkt)
            block_idx_before = builder.block_id
            emu.send(pkt)
            parity = builder.add(pkt)
            if parity is not None:
                parity_by_block[block_idx_before] = parity
                emu.send(parity)
        time.sleep(1.2)
        emu.stop()

        received_set = set(received_list)
        decoded_ok = 0
        reconstructed = 0
        for block_i in range(0, len(sent_full) // T.FEC_K):
            block = sent_full[block_i * T.FEC_K:(block_i + 1) * T.FEC_K]
            present = [pkt in received_set for pkt in block]
            decoded_ok += sum(present)
            missing_count = len(block) - sum(present)
            parity_pkt = parity_by_block.get(block_i)
            parity_survived = parity_pkt is not None and parity_pkt in received_set
            if missing_count == 1 and parity_survived:
                reconstructed += 1
                decoded_ok += 1

        raw_data_survival = sum(1 for pkt in sent_full if pkt in received_set) / len(sent_full)
        with_fec_survival = min(decoded_ok / len(sent_full), 1.0)
        link_level_survival = (emu.stats.delivered / emu.stats.sent) if emu.stats.sent else 0.0

        results.append({
            "configured_loss_rate": loss,
            "link_level_packet_survival": round(link_level_survival, 3),
            "raw_data_survival_rate_no_fec": round(raw_data_survival, 3),
            "survival_rate_with_fec": round(with_fec_survival, 3),
            "blocks_reconstructed": reconstructed,
            "total_blocks": len(sent_full) // T.FEC_K,
        })
    RESULTS["link_and_fec"] = results


def bench_localization():
    cam = CameraModel(fx=420, fy=420, cx=IMAGE_W / 2, cy=IMAGE_H / 2, height_m=1.2, pitch_deg=22)
    rover = RoverPose(x_m=0, y_m=0, heading_deg=0)
    test_points = [
        (IMAGE_W / 2, IMAGE_H / 2),
        (IMAGE_W / 2 - 80, IMAGE_H / 2 + 40),
        (IMAGE_W / 2 + 80, IMAGE_H / 2 - 20),
        (IMAGE_W / 2, IMAGE_H / 2 + 100),
    ]
    import math
    errors = []
    for px, py in test_points:
        res = pixel_to_landing_frame(px, py, IMAGE_W, IMAGE_H, cam, rover)
        # sanity re-derivation via independent closed-form for the center-pixel case
        errors.append(res["distance_m"])
    RESULTS["localization"] = {
        "camera_height_m": cam.height_m, "camera_pitch_deg": cam.pitch_deg,
        "center_pixel_distance_m": round(errors[0], 3),
        "expected_center_pixel_distance_m": round(cam.height_m / math.tan(math.radians(cam.pitch_deg)), 3),
        "distance_error_m": round(abs(errors[0] - cam.height_m / math.tan(math.radians(cam.pitch_deg))), 5),
        "sample_distances_m": [round(e, 2) for e in errors],
    }


def main():
    print("Running benchmarks...")
    bench_perception()
    bench_telemetry()
    bench_fec_and_link()
    bench_localization()

    with open("benchmark_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)

    print(json.dumps(RESULTS, indent=2))
    print("\nWritten to benchmark_results.json")


if __name__ == "__main__":
    main()
