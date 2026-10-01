#!/usr/bin/env python3

import sys
import copy
import contextlib
from pathlib import Path
import xml.etree.ElementTree as ET

import replay_index_dynamic_v1 as replay


# ============================================================
# DYNAMIC CONSISTENCY CHECK
#
# Replays the EXACT existing dynamic pipeline:
#
# command
#   -> 58-ms delay
#   -> asymmetric motor rate limiter
#   -> encoder
#   -> encoder-to-spool scale
#   -> tendon mechanics
#   -> passive MCP/PIP/DIP dynamics
#
# We change ONLY:
#
#   encoder-to-spool scale
#   tendon stiffness
#
# The original identified XML is never modified.
# ============================================================


SOURCE_MODEL = replay.MODEL_PATH

TENDON_NAME = replay.TENDON_NAME


CANDIDATES = [

    (
        "BASELINE_6p0um_1200",
        6.0,
        1200.0,
    ),

    (
        "A_5p1um_2100",
        5.1,
        2100.0,
    ),

    (
        "B_5p2um_2000",
        5.2,
        2000.0,
    ),

    (
        "C_5p3um_1900",
        5.3,
        1900.0,
    ),
]


RESULT_DIR = (
    replay.WS
    / "rh8d_digital_twin_tools/results/"
      "dynamic_candidate_comparison_v2"
)


def make_candidate_xml(
    source,
    output,
    stiffness,
):

    tree = ET.parse(
        source
    )

    root = tree.getroot()

    found = False


    for element in root.iter():

        if (
            element.attrib.get("name")
            == TENDON_NAME
        ):

            element.set(
                "stiffness",
                f"{stiffness:.8g}",
            )

            found = True

            break


    if not found:

        raise RuntimeError(
            f"Could not find tendon "
            f"'{TENDON_NAME}' in:\n"
            f"{source}"
        )


    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
    )


def main():

    if len(sys.argv) != 2:

        print()
        print(
            "Usage:"
        )

        print(
            "  python3 "
            "compare_index_dynamic_candidates_v2.py "
            "/path/to/original/free_space_rosbag"
        )

        sys.exit(1)


    bag = Path(
        sys.argv[1]
    ).expanduser().resolve()


    if not bag.exists():

        raise RuntimeError(
            f"Bag does not exist:\n{bag}"
        )


    if not SOURCE_MODEL.exists():

        raise RuntimeError(
            f"Source model does not exist:\n"
            f"{SOURCE_MODEL}"
        )


    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    candidate_models = []


    print()
    print("=" * 96)
    print(
        "RH8D INDEX — DYNAMIC TRANSMISSION "
        "CONSISTENCY CHECK V2"
    )
    print("=" * 96)

    print()
    print(
        f"Hardware bag:"
    )
    print(
        bag
    )

    print()
    print(
        f"Source model:"
    )
    print(
        SOURCE_MODEL
    )

    print()
    print(
        "Motor model is FROZEN:"
    )

    print(
        f"  delay        = "
        f"{replay.MOTOR_DELAY*1000:.1f} ms"
    )

    print(
        f"  closing rate = "
        f"{replay.CLOSING_RATE:.1f} ticks/s"
    )

    print(
        f"  opening rate = "
        f"{replay.OPENING_RATE:.1f} ticks/s"
    )


    # ========================================================
    # BUILD TEMPORARY MODELS
    # ========================================================

    for (
        label,
        scale_um,
        stiffness,
    ) in CANDIDATES:


        candidate_xml = (
            SOURCE_MODEL.parent
            /
            (
                "tmp_dynamic_"
                + label
                + ".xml"
            )
        )


        make_candidate_xml(
            SOURCE_MODEL,
            candidate_xml,
            stiffness,
        )


        candidate_models.append(
            (
                label,
                scale_um,
                stiffness,
                candidate_xml,
            )
        )


    # Save globals so nothing remains changed in Python state.
    original_model = (
        replay.MODEL_PATH
    )

    original_scale = (
        replay.METERS_PER_TICK
    )

    original_argv = copy.copy(
        sys.argv
    )


    try:

        for (
            label,
            scale_um,
            stiffness,
            candidate_xml,
        ) in candidate_models:


            print()
            print()
            print("#" * 96)
            print("#")
            print(
                f"# CANDIDATE: {label}"
            )
            print("#")
            print(
                f"# encoder/spool scale = "
                f"{scale_um:.3f} um/tick"
            )
            print(
                f"# tendon stiffness    = "
                f"{stiffness:.1f} N/m"
            )
            print("#")
            print("#" * 96)
            print()


            replay.MODEL_PATH = (
                candidate_xml
            )


            replay.METERS_PER_TICK = (
                scale_um
                * 1.0e-6
            )


            # replay.main() expects exactly one argument:
            # the original free-space rosbag directory.
            sys.argv = [
                "replay_index_dynamic_v1.py",
                str(bag),
            ]


            log_path = (
                RESULT_DIR
                /
                f"{label}.txt"
            )


            # ------------------------------------------------
            # Show replay output in terminal AND save it.
            # ------------------------------------------------

            class Tee:

                def __init__(
                    self,
                    *streams,
                ):
                    self.streams = streams

                def write(
                    self,
                    text,
                ):
                    for stream in self.streams:
                        stream.write(text)

                def flush(self):
                    for stream in self.streams:
                        stream.flush()


            with open(
                log_path,
                "w",
            ) as log_file:


                tee = Tee(
                    sys.__stdout__,
                    log_file,
                )


                with contextlib.redirect_stdout(
                    tee
                ):

                    replay.main()


            print()
            print(
                f"Candidate console log saved:"
            )

            print(
                log_path
            )


    finally:

        replay.MODEL_PATH = (
            original_model
        )

        replay.METERS_PER_TICK = (
            original_scale
        )

        sys.argv = (
            original_argv
        )


        # ----------------------------------------------------
        # Delete only our temporary candidate models.
        # Original XML is untouched.
        # ----------------------------------------------------

        for (
            _label,
            _scale,
            _stiffness,
            candidate_xml,
        ) in candidate_models:

            try:
                candidate_xml.unlink()
            except FileNotFoundError:
                pass


    print()
    print()
    print("=" * 96)
    print(
        "ALL FOUR DYNAMIC REPLAYS COMPLETE"
    )
    print("=" * 96)

    print()
    print(
        "Logs:"
    )

    print(
        RESULT_DIR
    )

    print()
    print(
        "Do NOT change the identified XML yet."
    )

    print("=" * 96)


if __name__ == "__main__":
    main()
