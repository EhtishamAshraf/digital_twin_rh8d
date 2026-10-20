#!/usr/bin/env python3
"""G43B: keep G39A MCP/PIP, transplant only G38C DIP route.
No geometry guesses, scalar fitting, passive changes or dynamic simulation.
"""
from copy import deepcopy
from pathlib import Path
import csv
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import audit_index_tendon_transmission_v35g12 as g12
import build_persistent_flexor_wrap_v35g38a as g38a
from rh8d_tabletop_stage_v1 import apply_tabletop_stage

HOME = Path.home() / "rh8d_results"
RECIPIENT = HOME / "v35g39a_cad_multiguide_flexor/g39a_cad_multiguide_flexor.xml"
DONOR = HOME / "v35g38c_valid_joint_pulley/g38c_valid_joint_pulley_flexor.xml"
OUT = HOME / "v35g43b_hybrid_dip"
OUT.mkdir(parents=True, exist_ok=True)
CANDIDATE = OUT / "g43b_hybrid_spatial.xml"
MAIN = "idx_flex_spatial"
START = "v35_p05_pip_child"
END = "v35_p08_distal_turn"

def spatial(root):
    for element in root.findall("./tendon/spatial"):
        if element.get("name") == MAIN:
            return element
    raise RuntimeError(f"Missing <spatial name='{MAIN}'>")

def anchor_idx(nodes, name):
    matching = [i for i, n in enumerate(nodes) if n.tag == "site" and n.get("site") == name]
    if len(matching) != 1:
        raise RuntimeError(f"Expected exactly one route anchor {name}, found {len(matching)}")
    return matching[0]

def locate_named(root, tag, name):
    matches = []
    for body in root.iter("body"):
        for child in body:
            if child.tag == tag and child.get("name") == name:
                matches.append((body, child))
    if len(matches) > 1:
        raise RuntimeError(f"Duplicate {tag} name {name}")
    return matches[0] if matches else None

def body_in(root, name):
    bodies = [b for b in root.iter("body") if b.get("name") == name]
    if len(bodies) != 1:
        raise RuntimeError(f"Body {name} is absent or ambiguous")
    return bodies[0]

def stage(model, data):
    apply_tabletop_stage(model, data, reset_to_qpos0=True)
    refs = g12.get_joint_refs(model)
    for k in ("mcp", "pip", "dip"):
        data.qpos[refs[k]["qadr"]] = 0.0
    mujoco.mj_forward(model, data)
    return refs

def tendon_len(model):
    data = mujoco.MjData(model)
    stage(model, data)
    return float(data.ten_length[model.tendon(MAIN).id])

for p in (RECIPIENT, DONOR):
    if not p.is_file():
        raise SystemExit(f"Required proven model not found: {p}")

R = ET.parse(RECIPIENT)
D = ET.parse(DONOR)
r, d = spatial(R.getroot()), spatial(D.getroot())
rnodes, dnodes = list(r), list(d)
ri, rj = anchor_idx(rnodes, START), anchor_idx(rnodes, END)

if not ri < rj:
    raise RuntimeError("Recipient DIP anchors out of order")

# G38C replaces some original CAD sites with pulley routing.
# Inspect the actual spatial tendon route.

print("\nG38C ACTIVE SPATIAL TENDON ROUTE")

for i, node in enumerate(dnodes):
    print(
        f"  {i:02d}: {node.tag:4s} "
        f"{node.get('geom') or node.get('site')} "
        f"sidesite={node.get('sidesite', '-')}"
    )

wrap_indices = [
    i for i, node in enumerate(dnodes)
    if node.tag == "geom"
]

dip_indices = [
    i for i in wrap_indices
    if "dip" in (dnodes[i].get("geom") or "").lower()
]

# If guide names lack joint labels, accept the third guide
# only for the verified three-pulley routing topology.

if len(dip_indices) != 1:
    if len(dip_indices) == 0 and len(wrap_indices) == 3:
        dip_indices = [wrap_indices[2]]
    else:
        raise RuntimeError(
            "STOP: G38C DIP wrap cannot be uniquely identified. "
            "Inspect the route printed above."
        )

k = dip_indices[0]

# A physical spatial wrapping guide must be bracketed
# by proper tendon routing sites.

if (
    k == 0
    or k + 1 >= len(dnodes)
    or dnodes[k-1].tag != "site"
    or dnodes[k+1].tag != "site"
):
    raise RuntimeError(
        "STOP: G38C DIP guide does not have two adjacent "
        "routing sites. No XML has been changed."
    )

# Copy ONLY the donor entrance site, DIP wrap and exit site.
# Preserve G39A's MCP/PIP routing and its p05/p08 anchors.

# The donor DIP wrap requires its original PIP-exit transition.
if k < 2 or dnodes[k-2].tag != "site":
    raise RuntimeError("STOP: donor PIP-exit site missing")

upstream = dnodes[k-2].get("site")

if upstream != "g38c_pip_exit":
    raise RuntimeError(
        f"STOP: unexpected donor upstream site {upstream}"
    )

old_site = locate_named(R.getroot(), "site", START)
new_site = locate_named(D.getroot(), "site", upstream)

if old_site is None or new_site is None:
    raise RuntimeError("STOP: routing boundary site missing")

if old_site[0].get("name") != new_site[0].get("name"):
    raise RuntimeError(
        "STOP: donor PIP exit and recipient p05 "
        "belong to different bodies"
    )

dip_nodes = [
    deepcopy(node)
    for node in dnodes[k-2:k+2]
]

print("\nSELECTED G38C DIP ROUTE:")

for node in dip_nodes:
    print(
        f"  {node.tag}: "
        f"{node.get('geom') or node.get('site')}"
    )
print("G39A DIP segment:", [(n.tag, n.get('geom') or n.get('site')) for n in rnodes[ri+1:rj]])
print("G38C DIP segment:", [(n.tag, n.get('geom') or n.get('site')) for n in dip_nodes])
if not any(n.tag == "geom" for n in dip_nodes):
    raise RuntimeError("G38C donor DIP span does not contain a wrapping geom. STOP.")

# Copy only needed DIP sites/geoms from G38C, into the same named body.
# Existing shared CAD sites MUST have identical local coordinates.
required = []
for n in dip_nodes:
    required.append((n.tag, n.get(n.tag)))
    if n.tag == "geom" and n.get("sidesite"):
        required.append(("site", n.get("sidesite")))
for tag, name in dict.fromkeys(required):
    if tag not in ("site", "geom"):
        raise RuntimeError(f"Unsupported donor tendon routing element: {tag}")
    found = locate_named(D.getroot(), tag, name)
    if found is None:
        raise RuntimeError(f"Donor definition missing: {tag} {name}")
    srcbody, srcelem = found
    existing = locate_named(R.getroot(), tag, name)
    if existing:
        recbody, recelem = existing
        if recbody.get("name") != srcbody.get("name"):
            raise RuntimeError(f"Shared routing item {name} changed owning body")
        for field in ("pos", "fromto", "quat", "size", "type"):
            if srcelem.get(field) != recelem.get(field):
                raise RuntimeError(f"Shared routing item {name} has different {field}; STOP")
    else:
        body_in(R.getroot(), srcbody.get("name")).append(deepcopy(srcelem))

# Remove unused G39A diagnostic DIP guide, if present (not MCP/PIP guides).
for name, tag in (("g39a_dip_cad_guide", "geom"), ("g39a_dip_cad_side", "site")):
    existing = locate_named(R.getroot(), tag, name)
    if existing:
        existing[0].remove(existing[1])

for node in rnodes[ri:rj]:
    r.remove(node)
insert_at = ri
for node in dip_nodes:
    r.insert(insert_at, node)
    insert_at += 1

# Keep the old elastic extension at the open/reference configuration.
old_model = mujoco.MjModel.from_xml_path(str(RECIPIENT))
old_len = tendon_len(old_model)
old_rest = float(r.get("springlength"))
# The intermediate file has the recipient rest length until re-referenced.
CANDIDATE.write_bytes(ET.tostring(R.getroot(), encoding="utf-8", xml_declaration=True))
new_model = mujoco.MjModel.from_xml_path(str(CANDIDATE))
new_len = tendon_len(new_model)
r.set("springlength", f"{old_rest + new_len - old_len:.12g}")
R.write(CANDIDATE, encoding="utf-8", xml_declaration=True)
model = mujoco.MjModel.from_xml_path(str(CANDIDATE))
data = mujoco.MjData(model)
refs = stage(model, data)
tid = model.tendon(MAIN).id

print("\nG43B HYBRID GEOMETRY AUDIT")
print("Open tendon length change: %.4f mm" % (1000*(new_len-old_len)))
print("MCP/PIP: G39A original; DIP: transplanted from G38C")
print("Actuators, joints, friction, stiffness: unchanged")
rows, failed = [], []
for joint in ("mcp", "pip", "dip"):
    values = []
    for deg in range(0, 86, 5):
        stage(model, data)
        data.qpos[refs[joint]["qadr"]] = math.radians(deg)
        mujoco.mj_forward(model, data)
        J = g38a.tendon_jacobian_row(model, data, tid)
        value = 1000.0 * float(J[refs[joint]["dadr"]])
        rows.append({"test": f"single_{joint}", "angle": deg,
                     "mcp": "", "pip": "", "dip": "", "J_mm_rad": value})
        values.append(value)
        if not np.isfinite(value) or value >= -1.0:
            failed.append((joint, deg, value))
    print(f"{joint.upper()}: J min/max = {min(values):+.3f} / {max(values):+.3f} mm/rad")

# Recheck donor at measured combined joint configurations, not just isolated joints.
video13 = [(0,0,0), (0,0.05,0), (0,0.5,0), (0,1.0,0), (16.952,.968,0), (39.472,0,0), (57.291,3.654,0),
           (73.632,4.667,2.012), (73.632,27.119,3.190),
           (77.347,48.581,7.810), (80.074,70.872,0)]
print("\nVIDEO13 joint combined-state checks (J in mm/rad)")
for angles in video13:
    stage(model, data)
    for joint, deg in zip(("mcp","pip","dip"), angles):
        data.qpos[refs[joint]["qadr"]] = math.radians(deg)
    mujoco.mj_forward(model, data)
    J = g38a.tendon_jacobian_row(model, data, tid)
    vals = [1000.0 * float(J[refs[k]["dadr"]]) for k in ("mcp","pip","dip")]
    print("%6.1f/%6.1f/%6.1f: %+6.2f %+6.2f %+6.2f" % (*angles, *vals))
    if any(not np.isfinite(x) or x >= -1.0 for x in vals):
        failed.append((angles, vals))
    rows.append({"test":"video13", "angle":"", "mcp":angles[0],"pip":angles[1],
                 "dip":angles[2],"J_mm_rad":"/".join(f"{v:.5f}" for v in vals)})

with (OUT / "g43b_audit.csv").open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print("\nG43B DECISION:", "PASS geometry only" if not failed else "FAIL geometry")
if failed:
    print("Failures:", failed[:15])
print("Candidate XML:", CANDIDATE)
print("Audit CSV:", OUT / "g43b_audit.csv")
if failed:
    raise SystemExit(2)
