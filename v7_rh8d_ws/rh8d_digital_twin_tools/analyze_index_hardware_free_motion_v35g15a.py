#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import scan_index_contact_friction_v35g7 as g7


# ================================================================
# G15A — HARDWARE FREE-MOTION TRANSMISSION AUDIT
#
# Uses EXISTING hardware data only.
#
# Goal:
#   characterize real MCP/PIP/DIP behavior BEFORE first object contact.
#
# No MuJoCo parameter fitting.
# No contact fitting.
# ================================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "bag",
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            Path.home()
            / "rh8d_results"
            / "v35g15a_hardware_free_motion"
        ),
    )

    args = parser.parse_args()

    bag = args.bag.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Reuse the exact bag parser already validated by previous experiments.
    g7.b2.v30.v24.ab.patch_replay_modules()

    contact_replay = (
        g7.b2.v30.v24.base.contact_replay
    )

    # ------------------------------------------------------------
    # Load the same synchronized hardware dataset used by replay.
    # ------------------------------------------------------------

    hw = contact_replay.load_data(
        bag
    )

    # Print available keys first for traceability.
    print()
    print("=" * 100)
    print("G15A — HARDWARE FREE-MOTION AUDIT")
    print("=" * 100)
    print()
    print("Available hardware fields:")

    if isinstance(hw, dict):
        for key in sorted(hw.keys()):
            try:
                shape = np.asarray(hw[key]).shape
            except Exception:
                shape = "?"
            print(f"  {key}: {shape}")
    else:
        print(type(hw))

    # ------------------------------------------------------------
    # The existing replay code may expose fields under slightly
    # different names. Resolve common possibilities explicitly.
    # ------------------------------------------------------------

    def resolve(names):
        for name in names:
            if isinstance(hw, dict) and name in hw:
                return np.asarray(hw[name])
        return None

    t = resolve([
        "time",
        "times",
        "t",
        "bag_time",
        "timestamp",
    ])

    encoder = resolve([
        "encoder",
        "encoder_tick",
        "ticks",
        "motor_encoder",
    ])

    mcp = resolve([
        "mcp_deg",
        "mcp",
        "index_mcp_deg",
    ])

    pip = resolve([
        "pip_deg",
        "pip",
        "index_pip_deg",
    ])

    dip = resolve([
        "dip_deg",
        "dip",
        "index_dip_deg",
    ])

    if any(x is None for x in [t, encoder, mcp, pip, dip]):
        raise RuntimeError(
            "\nCould not automatically resolve one or more hardware fields.\n"
            "The printed field list above is enough to patch the names.\n"
            "Send me that output if this happens."
        )

    t = np.asarray(t, dtype=float).reshape(-1)
    encoder = np.asarray(encoder, dtype=float).reshape(-1)
    mcp = np.asarray(mcp, dtype=float).reshape(-1)
    pip = np.asarray(pip, dtype=float).reshape(-1)
    dip = np.asarray(dip, dtype=float).reshape(-1)

    n = min(
        len(t),
        len(encoder),
        len(mcp),
        len(pip),
        len(dip),
    )

    t = t[:n]
    encoder = encoder[:n]
    mcp = mcp[:n]
    pip = pip[:n]
    dip = dip[:n]

    # ------------------------------------------------------------
    # H1 first middle contact occurred at approximately the encoder
    # range found in previous G6/G12 runs (~870 ticks).
    #
    # We deliberately stop before contact.
    # ------------------------------------------------------------

    FREE_ENCODER_MAX = 840.0

    mask = (
        np.isfinite(t)
        &
        np.isfinite(encoder)
        &
        np.isfinite(mcp)
        &
        np.isfinite(pip)
        &
        np.isfinite(dip)
        &
        (encoder <= FREE_ENCODER_MAX)
    )

    if np.count_nonzero(mask) < 10:
        raise RuntimeError(
            "Too few free-motion samples below encoder=840. "
            "Do not change the threshold yet; send me the field output."
        )

    tf = t[mask]
    ef = encoder[mask]
    mf = mcp[mask]
    pf = pip[mask]
    df = dip[mask]

    # Sort by time.
    order = np.argsort(tf)

    tf = tf[order]
    ef = ef[order]
    mf = mf[order]
    pf = pf[order]
    df = df[order]

    # ------------------------------------------------------------
    # Normalize joint displacement from beginning of this segment.
    # ------------------------------------------------------------

    dm = mf - mf[0]
    dp = pf - pf[0]
    dd = df - df[0]
    de = ef - ef[0]

    # ------------------------------------------------------------
    # Encoder-domain slopes.
    # deg / encoder tick
    # ------------------------------------------------------------

    def fit_slope(x, y):

        good = (
            np.isfinite(x)
            &
            np.isfinite(y)
        )

        if np.count_nonzero(good) < 3:
            return np.nan, np.nan

        c = np.polyfit(
            x[good],
            y[good],
            1,
        )

        pred = np.polyval(
            c,
            x[good],
        )

        ss_res = np.sum(
            (y[good] - pred) ** 2
        )

        ss_tot = np.sum(
            (
                y[good]
                -
                np.mean(y[good])
            ) ** 2
        )

        r2 = (
            1.0
            -
            ss_res
            /
            max(ss_tot, 1e-12)
        )

        return float(c[0]), float(r2)

    mcp_slope, mcp_r2 = fit_slope(
        de,
        dm,
    )

    pip_slope, pip_r2 = fit_slope(
        de,
        dp,
    )

    dip_slope, dip_r2 = fit_slope(
        de,
        dd,
    )

    # ------------------------------------------------------------
    # Local motion ratios.
    # ------------------------------------------------------------

    if len(tf) >= 3:

        m_vel = np.gradient(
            mf,
            tf,
        )

        p_vel = np.gradient(
            pf,
            tf,
        )

        d_vel = np.gradient(
            df,
            tf,
        )

    else:

        m_vel = np.full_like(
            mf,
            np.nan,
        )

        p_vel = np.full_like(
            pf,
            np.nan,
        )

        d_vel = np.full_like(
            df,
            np.nan,
        )

    dip_over_pip = (
        np.abs(d_vel)
        /
        np.maximum(
            np.abs(p_vel),
            1e-6,
        )
    )

    print()
    print("FREE-MOTION REGION")
    print("==================")
    print(
        f"encoder range = "
        f"{ef[0]:.2f} -> {ef[-1]:.2f}"
    )
    print(
        f"time range    = "
        f"{tf[0]:.3f} -> {tf[-1]:.3f} s"
    )
    print(
        f"samples       = "
        f"{len(tf)}"
    )

    print()
    print("TOTAL FREE-MOTION JOINT CHANGE")
    print("==============================")
    print(
        f"MCP = {dm[-1]:+.3f} deg"
    )
    print(
        f"PIP = {dp[-1]:+.3f} deg"
    )
    print(
        f"DIP = {dd[-1]:+.3f} deg"
    )

    print()
    print("ANGLE PER ENCODER TICK")
    print("======================")
    print(
        f"MCP = {mcp_slope:+.8f} deg/tick "
        f"(R2={mcp_r2:.4f})"
    )
    print(
        f"PIP = {pip_slope:+.8f} deg/tick "
        f"(R2={pip_r2:.4f})"
    )
    print(
        f"DIP = {dip_slope:+.8f} deg/tick "
        f"(R2={dip_r2:.4f})"
    )

    print()
    print("MEDIAN FREE-MOTION SPEED")
    print("========================")
    print(
        f"MCP = "
        f"{np.median(np.abs(m_vel)):.3f} deg/s"
    )
    print(
        f"PIP = "
        f"{np.median(np.abs(p_vel)):.3f} deg/s"
    )
    print(
        f"DIP = "
        f"{np.median(np.abs(d_vel)):.3f} deg/s"
    )

    print()
    print(
        "median |DIP velocity| / |PIP velocity| = "
        f"{np.median(dip_over_pip):.3f}"
    )

    # ------------------------------------------------------------
    # Save timeseries.
    # ------------------------------------------------------------

    csv_path = (
        out_dir
        / "hardware_free_motion.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "time_s",
            "encoder_tick",
            "mcp_deg",
            "pip_deg",
            "dip_deg",
            "delta_mcp_deg",
            "delta_pip_deg",
            "delta_dip_deg",
            "mcp_vel_deg_s",
            "pip_vel_deg_s",
            "dip_vel_deg_s",
            "dip_over_pip_speed",
        ])

        for i in range(len(tf)):

            writer.writerow([
                tf[i],
                ef[i],
                mf[i],
                pf[i],
                df[i],
                dm[i],
                dp[i],
                dd[i],
                m_vel[i],
                p_vel[i],
                d_vel[i],
                dip_over_pip[i],
            ])

    with (
        out_dir
        / "summary.txt"
    ).open("w") as f:

        f.write(
            "G15A hardware free-motion audit\n"
        )
        f.write(
            "================================\n\n"
        )

        f.write(
            f"encoder range: "
            f"{ef[0]:.4f} -> {ef[-1]:.4f}\n"
        )

        f.write(
            f"MCP total change: "
            f"{dm[-1]:+.6f} deg\n"
        )

        f.write(
            f"PIP total change: "
            f"{dp[-1]:+.6f} deg\n"
        )

        f.write(
            f"DIP total change: "
            f"{dd[-1]:+.6f} deg\n\n"
        )

        f.write(
            f"MCP slope: "
            f"{mcp_slope:+.10f} deg/tick\n"
        )

        f.write(
            f"PIP slope: "
            f"{pip_slope:+.10f} deg/tick\n"
        )

        f.write(
            f"DIP slope: "
            f"{dip_slope:+.10f} deg/tick\n"
        )

        f.write(
            f"median DIP/PIP speed ratio: "
            f"{np.median(dip_over_pip):.6f}\n"
        )

    print()
    print(
        f"Outputs saved to:\n  {out_dir}"
    )


if __name__ == "__main__":
    main()
