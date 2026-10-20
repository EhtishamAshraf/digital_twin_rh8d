#!/usr/bin/env python3
"""Controlled 2x2 contact solref experiment for the established V43 replay."""
import csv
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import matplotlib.pyplot as plt

import test_v43_pd_compliant_coupling as v43

WS = Path.home() / (
    'Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/'
    '2-MuJoCo/digital_twin/v7_rh8d_ws'
)
MJCF = WS / 'src/ur5e_mujoco_description/mjcf'
SOURCE = MJCF / 'tmp_v43_k750_r20.xml'
OUT = WS / 'rh8d_digital_twin_tools/results/v43_contact_solref_2x2'
OUT.mkdir(parents=True, exist_ok=True)

MIDDLE = {'collision_idx_mid_mesh', 'collision_idx_mid'}
TIP = {'collision_idx_tip'}
HW = np.array([14.5355, 23.8110, 9.3270])
JOINTS = [
    'Index_Proximal--palm_assembly:1',
    'Index_Middle--Index_Proximal',
    'Index_Distal--Index_Middle',
]
CASES = [
    ('K1000_D20', 1000.0, 20.0),
    ('K1000_D1', 1000.0, 1.0),
    ('K14000_D20', 14000.0, 20.0),
    ('K14000_D1', 14000.0, 1.0),
]


def geom_id(model, name):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        raise RuntimeError(f'Missing geom: {name}')
    return int(gid)


def make_xml(label, k, b):
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    fixture = root.find(".//geom[@name='index_contact_fixture']")
    if fixture is None:
        raise RuntimeError('Missing fixture geom in XML')
    fixture.set('solref', f'-{k:.12g} -{b:.12g}')

    # Make mesh/texture assets independent of the XML output directory.
    compiler = root.find('compiler')
    if compiler is None:
        compiler = ET.Element('compiler')
        root.insert(0, compiler)
    for key in ('meshdir', 'texturedir'):
        value = compiler.get(key)
        if value and not Path(value).is_absolute():
            compiler.set(key, str((MJCF / value).resolve()))

    path = OUT / f'{label}.xml'
    tree.write(path, encoding='unicode', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    gid = geom_id(model, 'index_contact_fixture')
    if not np.allclose(model.geom_solref[gid], [-k, -b]):
        raise RuntimeError(f'Compiled fixture mismatch: {label}')
    return path


def run(label, xml, k, b):
    rows = []
    state = {}
    actual_step = mujoco.mj_step

    def record(model, data, *args, **kwargs):
        ret = actual_step(model, data, *args, **kwargs)
        if not (4.0 <= data.time <= 8.402):
            return ret
        if not state:
            fixture = geom_id(model, 'index_contact_fixture')
            state['fixture'] = fixture
            state['solref'] = np.array(model.geom_solref[fixture], copy=True)
            if not np.allclose(state['solref'], [-k, -b]):
                raise RuntimeError(
                    f'{label}: runtime fixture solref {state["solref"]}, '
                    f'expected {[-k, -b]}'
                )
            state['qaddr'] = []
            for name in JOINTS:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise RuntimeError(f'Missing joint {name}')
                state['qaddr'].append(int(model.jnt_qposadr[jid]))

        fm, ft, samples = 0.0, 0.0, 0
        for i in range(data.ncon):
            contact = data.contact[i]
            a, bb = int(contact.geom1), int(contact.geom2)
            if state['fixture'] not in (a, bb):
                continue
            other = bb if a == state['fixture'] else a
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
            if name not in MIDDLE | TIP:
                continue
            samples += 1
            # Effective parameters on the actual contact constraint.
            if not np.allclose(contact.solref, [-k, -b]):
                raise RuntimeError(
                    f'{label}: effective contact solref {contact.solref} '
                    f'differs from {[-k, -b]}'
                )
            f = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, i, f)
            normal = max(0.0, float(f[0]))
            if name in MIDDLE:
                fm += normal
            else:
                ft += normal

        if samples:
            state['contact_samples'] = state.get('contact_samples', 0) + samples
        angles = np.degrees(np.asarray(data.qpos[state['qaddr']]))
        err = float(np.sqrt(np.mean((angles-HW)**2)))
        rows.append({
            'time_s': float(data.time),
            'middle_normal_N': fm,
            'tip_normal_N': ft,
            'total_normal_N': fm + ft,
            'mcp_deg': float(angles[0]),
            'pip_deg': float(angles[1]),
            'dip_deg': float(angles[2]),
            'target_angle_rmse_deg': err,
        })
        return ret

    mujoco.mj_step = record
    try:
        result = v43.run_contact(xml, label)
    finally:
        mujoco.mj_step = actual_step

    if len(rows) < 3000 or not state.get('contact_samples'):
        raise RuntimeError(f'{label}: incomplete replay/capture: {len(rows)} steps')
    csv_path = OUT / f'{label}_trace.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    t = np.asarray([r['time_s'] for r in rows])
    fn = np.asarray([r['total_normal_N'] for r in rows])
    mid = np.asarray([r['middle_normal_N'] for r in rows])
    tip = np.asarray([r['tip_normal_N'] for r in rows])
    er = np.asarray([r['target_angle_rmse_deg'] for r in rows])
    dt = float(np.median(np.diff(t)))
    hold = (t >= 6.2) & (t <= 8.0)
    pi = int(np.argmax(fn))
    summary = dict(
        case=label, k_Npm=k, b_Nspm=b,
        reported_contact_rmse_deg=float(result['rmse_deg']),
        best_instant_rmse_deg=float(er.min()),
        terminal_rmse_deg=float(er[-1]),
        peak_normal_N=float(fn[pi]),
        peak_at_s=float(t[pi]),
        peak_middle_N=float(mid[pi]),
        peak_tip_N=float(tip[pi]),
        p95_normal_N=float(np.percentile(fn, 95)),
        above_2N_ms=float((fn>2).sum()*dt*1000),
        above_50N_ms=float((fn>50).sum()*dt*1000),
        normal_impulse_Ns=float(np.trapz(fn, t)),
        fraction_any_load_6p2_to_8=float(np.mean(((mid>0.02)|(tip>0.02))[hold])),
        fraction_both_load_6p2_to_8=float(np.mean(((mid>0.02)&(tip>0.02))[hold])),
        final_mcp_deg=float(rows[-1]['mcp_deg']),
        final_pip_deg=float(rows[-1]['pip_deg']),
        final_dip_deg=float(rows[-1]['dip_deg']),
    )
    return summary, t, fn, er


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if os.environ.get('RH8D_PRESERVE_XML_SOLREF') not in (None, '', '1'):
        print('Note: forcing XML solref preservation for this experiment.')
    old_flag = os.environ.get('RH8D_PRESERVE_XML_SOLREF')
    os.environ['RH8D_PRESERVE_XML_SOLREF'] = '1'
    output = []
    curves = {}
    try:
        for name, k, b in CASES:
            xml = make_xml(name, k, b)
            print(f'\nRunning {name}: solref=[-{k:g}, -{b:g}]')
            summary, t, fn, err = run(name, xml, k, b)
            output.append(summary)
            curves[name] = (t, fn, err)
            print(f'RMSE={summary["reported_contact_rmse_deg"]:.4f} deg, '
                  f'peak={summary["peak_normal_N"]:.3f} N, '
                  f'>50 N={summary["above_50N_ms"]:.1f} ms, '
                  f'finalRMSE={summary["terminal_rmse_deg"]:.3f} deg')
    finally:
        if old_flag is None:
            os.environ.pop('RH8D_PRESERVE_XML_SOLREF', None)
        else:
            os.environ['RH8D_PRESERVE_XML_SOLREF'] = old_flag

    baseline = output[0]
    if abs(baseline['reported_contact_rmse_deg'] - 0.7042798331155976) > 1e-4:
        raise RuntimeError('K1000_D20 did NOT reproduce the historical V43 result.')

    path = OUT / 'comparison.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for name, (t, fn, err) in curves.items():
        axes[0].plot(t, fn + 1e-4, label=name)
        axes[1].plot(t, err, label=name)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Total normal force + 0.0001 (N)')
    axes[1].set_ylabel('Angle RMSE (degrees)')
    axes[1].set_xlabel('Simulation time (s)')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    image_file = OUT / 'force_and_angle_comparison.png'
    fig.savefig(image_file, dpi=150)
    plt.close(fig)

    print('\n' + '='*110)
    print('CONTACT SOLREF 2x2: FINAL COMPARISON')
    print('='*110)
    print(f'{"case":17} {"fit RMSE":>10} {"final RMSE":>11} {"peak N":>10} '
          f'{">50N ms":>9} {"both loaded":>12}')
    for r in output:
        print(f'{r["case"]:17} {r["reported_contact_rmse_deg"]:10.3f} '
              f'{r["terminal_rmse_deg"]:11.3f} {r["peak_normal_N"]:10.3f} '
              f'{r["above_50N_ms"]:9.1f} '
              f'{r["fraction_both_load_6p2_to_8"]:12.3f}')
    print('\nCSV:', path)
    print('Plot:', image_file)
    print('Source and replay files unchanged by this script.')

if __name__ == '__main__':
    main()
