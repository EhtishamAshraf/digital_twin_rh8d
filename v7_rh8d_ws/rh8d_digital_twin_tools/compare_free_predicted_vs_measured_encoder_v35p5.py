#!/usr/bin/env python3

from pathlib import Path
import contextlib
import io
import math
import re
import sys

import numpy as np

import replay_index_dynamic_v1 as predicted
import replay_index_dynamic_measured_encoder_v2 as measured


WS = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/"
    "digital_twin/v7_rh8d_ws"
).expanduser().resolve()

MODEL = (
    WS
    / "src/ur5e_mujoco_description/mjcf"
    / "ur5e_rh8d_index_cad_spatial_v35i1_best.xml"
)

FREE_BAG = Path(
    "~/Desktop/Robotics_uclv/03_PROJECTS/P2_rh8d_hardware/"
    "rh8d_robot_control/v1_rh8dR_ws/"
    "rh8d_dashboard_data/rosbags/"
    "rh8d_bag_20260910_103316"
).expanduser().resolve()

OUTDIR = (
    WS
    / "rh8d_digital_twin_tools/results"
    / "v35p5_free_predicted_vs_measured_encoder"
)

SCALE = 5.10e-6


def value(text, pattern):

    m = re.search(
        pattern,
        text,
        re.MULTILINE | re.IGNORECASE,
    )

    if m is None:
        return math.nan

    return float(
        m.group(1)
    )


def parse_reference_table(text):

    rows = []

    pat = re.compile(
        r"^\s*"
        r"(closing|opening)\s+"
        r"([0-9.]+)\s+"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s*$",
        re.MULTILINE,
    )


    for m in pat.finditer(
        text
    ):

        rows.append({
            "phase":
                m.group(1),

            "target":
                float(
                    m.group(2)
                ),

            # NOTE:
            #
            # In BOTH runs this column remains the prediction
            # of the motor model.
            #
            # In the measured-mechanics run it is NOT the
            # encoder used to drive the finger.
            "predicted_motor":
                float(
                    m.group(3)
                ),

            "hw_mcp":
                float(
                    m.group(4)
                ),

            "sim_mcp":
                float(
                    m.group(5)
                ),

            "hw_pip":
                float(
                    m.group(6)
                ),

            "sim_pip":
                float(
                    m.group(7)
                ),
        })


    return rows


def phase_rmse(
    rows,
    phase=None,
):

    use = [
        r
        for r in rows
        if (
            phase is None
            or
            r["phase"] == phase
        )
    ]


    if not use:

        return math.nan, math.nan, math.nan


    em = np.asarray(
        [
            r["sim_mcp"]
            -
            r["hw_mcp"]
            for r in use
        ],
        dtype=float,
    )

    ep = np.asarray(
        [
            r["sim_pip"]
            -
            r["hw_pip"]
            for r in use
        ],
        dtype=float,
    )


    mcp = float(
        np.sqrt(
            np.mean(
                em * em
            )
        )
    )

    pip = float(
        np.sqrt(
            np.mean(
                ep * ep
            )
        )
    )

    combined = float(
        np.sqrt(
            (
                mcp * mcp
                +
                pip * pip
            )
            / 2.0
        )
    )


    return (
        mcp,
        pip,
        combined,
    )


def run_module(
    module,
    label,
):

    old_argv = list(
        sys.argv
    )

    old_model = getattr(
        module,
        "MODEL_PATH",
        None,
    )

    old_scale = getattr(
        module,
        "METERS_PER_TICK",
        None,
    )

    old_k = getattr(
        module,
        "TENDON_STIFFNESS",
        None,
    )


    try:

        if hasattr(
            module,
            "MODEL_PATH",
        ):

            module.MODEL_PATH = MODEL


        if hasattr(
            module,
            "METERS_PER_TICK",
        ):

            module.METERS_PER_TICK = SCALE


        # Some historical versions expose a global tendon
        # stiffness, while others read it from the candidate XML.
        #
        # The V35I1 XML itself contains the 2100-N/m baseline.
        if hasattr(
            module,
            "TENDON_STIFFNESS",
        ):

            module.TENDON_STIFFNESS = 2100.0


        sys.argv = [
            module.__file__,
            str(
                FREE_BAG
            ),
        ]


        stream = io.StringIO()

        with contextlib.redirect_stdout(
            stream
        ):

            module.main()


        text = stream.getvalue()


    finally:

        sys.argv = old_argv


        if (
            old_model
            is not None
            and
            hasattr(
                module,
                "MODEL_PATH",
            )
        ):

            module.MODEL_PATH = old_model


        if (
            old_scale
            is not None
            and
            hasattr(
                module,
                "METERS_PER_TICK",
            )
        ):

            module.METERS_PER_TICK = old_scale


        if (
            old_k
            is not None
            and
            hasattr(
                module,
                "TENDON_STIFFNESS",
            )
        ):

            module.TENDON_STIFFNESS = old_k


    (OUTDIR / f"{label}.log").write_text(
        text
    )


    rows = parse_reference_table(
        text
    )


    return {
        "label":
            label,

        "text":
            text,

        "rows":
            rows,

        "motor_rmse":
            value(
                text,
                r"encoder RMSE\s*=\s*"
                r"([-+0-9.eE]+)\s*ticks",
            ),

        "mcp_reported":
            value(
                text,
                r"MCP RMSE\s*=\s*"
                r"([-+0-9.eE]+)\s*deg",
            ),

        "pip_reported":
            value(
                text,
                r"PIP RMSE\s*=\s*"
                r"([-+0-9.eE]+)\s*deg",
            ),
    }


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    if not MODEL.exists():

        raise RuntimeError(
            f"Missing model:\n{MODEL}"
        )


    if not (
        FREE_BAG
        / "metadata.yaml"
    ).exists():

        raise RuntimeError(
            f"Missing free-space bag:\n{FREE_BAG}"
        )


    print()
    print("=" * 120)
    print(
        "V35P5 — FREE-SPACE MOTOR/MECHANICS DECOUPLING A/B"
    )
    print("=" * 120)

    print(
        f"MODEL:\n{MODEL}"
    )

    print(
        f"\nFREE BAG:\n{FREE_BAG}"
    )

    print()
    print(
        "A = predicted encoder drives mechanics"
    )

    print(
        "B = measured encoder drives mechanics"
    )

    print()
    print(
        "Motor model remains active and is scored in BOTH."
    )


    print()
    print(
        "Running A..."
    )

    A = run_module(
        predicted,
        "A_predicted_encoder_mechanics",
    )


    print(
        "Running B..."
    )

    B = run_module(
        measured,
        "B_measured_encoder_mechanics",
    )


    print()
    print("=" * 120)
    print("MOTOR LAYER")
    print("=" * 120)

    print(
        f"A encoder RMSE = "
        f"{A['motor_rmse']:.3f} ticks"
    )

    print(
        f"B encoder RMSE = "
        f"{B['motor_rmse']:.3f} ticks"
    )

    print()
    print(
        "These SHOULD be essentially identical."
    )

    print(
        "If not, the measured-mechanics patch accidentally "
        "changed the motor model."
    )


    print()
    print("=" * 120)
    print("FINGER LAYER — REPORTED BY ORIGINAL EVALUATOR")
    print("=" * 120)

    print(
        f"A MCP/PIP = "
        f"{A['mcp_reported']:.3f} / "
        f"{A['pip_reported']:.3f} deg"
    )

    print(
        f"B MCP/PIP = "
        f"{B['mcp_reported']:.3f} / "
        f"{B['pip_reported']:.3f} deg"
    )


    print()
    print("=" * 120)
    print("FINGER LAYER — PHASE-SPLIT RMSE")
    print("=" * 120)


    for phase in [
        "closing",
        "opening",
        None,
    ]:

        name = (
            "ALL"
            if phase is None
            else phase.upper()
        )


        am, ap, ac = phase_rmse(
            A["rows"],
            phase,
        )

        bm, bp, bc = phase_rmse(
            B["rows"],
            phase,
        )


        print()
        print(
            f"{name}:"
        )

        print(
            "  A predicted encoder:"
        )

        print(
            f"    MCP={am:.3f}  "
            f"PIP={ap:.3f}  "
            f"combined={ac:.3f} deg"
        )

        print(
            "  B measured encoder:"
        )

        print(
            f"    MCP={bm:.3f}  "
            f"PIP={bp:.3f}  "
            f"combined={bc:.3f} deg"
        )

        print(
            f"  delta combined B-A = "
            f"{bc-ac:+.3f} deg"
        )


    print()
    print("=" * 120)
    print("REFERENCE-FRAME DIFFERENCE CAUSED ONLY BY ENCODER INPUT")
    print("=" * 120)

    if (
        len(
            A["rows"]
        )
        !=
        len(
            B["rows"]
        )
    ):

        print(
            "WARNING: reference table row count differs."
        )

    else:

        print(
            "phase    target     "
            "A_MCP    B_MCP   delta     "
            "A_PIP    B_PIP   delta"
        )

        print("-" * 90)


        for a, b in zip(
            A["rows"],
            B["rows"],
        ):

            print(
                f"{a['phase']:7s} "
                f"{a['target']:7.0f}   "
                f"{a['sim_mcp']:7.2f} "
                f"{b['sim_mcp']:7.2f} "
                f"{b['sim_mcp']-a['sim_mcp']:+7.2f}   "
                f"{a['sim_pip']:7.2f} "
                f"{b['sim_pip']:7.2f} "
                f"{b['sim_pip']-a['sim_pip']:+7.2f}"
            )


    print()
    print("=" * 120)
    print("INTERPRETATION GATE")
    print("=" * 120)


    _, _, Aall = phase_rmse(
        A["rows"],
        None,
    )

    _, _, Ball = phase_rmse(
        B["rows"],
        None,
    )


    improvement = (
        Aall
        -
        Ball
    )


    if improvement > 1.0:

        print(
            "MEASURED ENCODER MATERIALLY IMPROVES "
            "FREE-SPACE MECHANICS."
        )

        print(
            "Previous finger parameters likely absorbed "
            "part of the motor-model error."
        )

        print(
            "Next: re-identify free-space mechanics using "
            "measured encoder as the mechanical boundary."
        )

    elif improvement < -1.0:

        print(
            "MEASURED ENCODER MATERIALLY WORSENS "
            "FREE-SPACE MECHANICS."
        )

        print(
            "This is still important: the current mechanics "
            "were evidently calibrated around motor-model error."
        )

        print(
            "Next: re-identify free-space mechanics using "
            "the measured encoder boundary."
        )

    else:

        print(
            "MOTOR-MODEL ERROR HAS ONLY A SMALL EFFECT "
            "ON THE FREE-SPACE FINGER RMSE."
        )

        print(
            "The old free-space mechanical calibration is "
            "less contaminated than suspected."
        )

        print(
            "Next: audit the remaining one-step spool drift "
            "and current spatial/passive model before refitting."
        )


    print()
    print(
        f"Full logs:\n{OUTDIR}"
    )

    print("=" * 120)


if __name__ == "__main__":
    main()
