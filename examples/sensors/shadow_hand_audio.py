"""
Shadow Hand "finger drums": tap colored blocks of different materials and hear them.

A palm-down Shadow Hand with a fixed base hovers over four blocks -- red / yellow / green / blue -- one under each of
the index / middle / ring / pinky fingertips, each block a different vibroacoustic material (metal / glass / wood /
ceramic). Press 1/2/3/4 to IK-tap that finger straight down onto its block; the ContactAudio sensor on the fingertip
synthesizes the material's impact sound, while a quiet ActuationSource voices the finger motors. A SpatialAudio
microphone renders both through the air, written to ``<out>.wav`` + ``<out>_spectrogram.png``.

    python examples/sensors/shadow_hand_audio.py -v                       # interactive (keys 1/2/3/4)
    python examples/sensors/shadow_hand_audio.py --cpu -t 4.0 -o hand.wav # headless: scripted taps

Audio is the source/receiver pipeline: ContactAudio (contact mics) + ActuationSource (motor source) -> AudioManager
registry -> SpatialAudio (airborne mic).
"""

import argparse
import os

import numpy as np
from contact_audio_teleop import write_spectrogram, write_wav

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

DT = 0.005
AUDIO_SUBSTEPS = 80  # 16 kHz
N_MODES = 4

BLOCK_SIZE = 0.035
DROP = 0.055  # how far below the resting fingertip its block sits (the tap depth)
TAP_STEPS = 45  # how long a keypress holds the finger down

# finger label -> (fingertip link name, key, block color rgba)
FINGERS = [
    ("index", "index_finger_distal", Key._1, (0.85, 0.2, 0.2, 1.0)),
    ("middle", "middle_finger_distal", Key._2, (0.9, 0.8, 0.2, 1.0)),
    ("ring", "ring_finger_distal", Key._3, (0.25, 0.75, 0.35, 1.0)),
    ("pinky", "little_finger_distal", Key._4, (0.3, 0.5, 0.95, 1.0)),
]

# Per-block vibroacoustic material (the struck block). metal rings long & bright; glass higher & longer; wood low &
# dead with a grainy texture; ceramic mid with a sharp click.
MATERIALS = {
    "index": gs.sensors.ContactAudioProperties(  # red = metal
        modal_freqs=(1200.0, 2450.0, 3700.0),
        modal_decays=(3.0, 4.5, 6.0),
        modal_gains=(1.0, 0.7, 0.4),
        impact_gain=1.6,
        impact_threshold=0.4,
        contact_damping=4000.0,
        accel_noise_gain=0.6,
    ),
    "middle": gs.sensors.ContactAudioProperties(  # yellow = glass
        modal_freqs=(2000.0, 4100.0, 6300.0),
        modal_decays=(1.5, 2.5, 3.5),
        modal_gains=(1.0, 0.6, 0.3),
        impact_gain=2.2,
        impact_threshold=0.4,
        contact_damping=6000.0,
        accel_noise_gain=0.8,
    ),
    "ring": gs.sensors.ContactAudioProperties(  # green = wood
        modal_freqs=(120.0, 300.0, 620.0),
        modal_decays=(55.0, 80.0, 110.0),
        modal_gains=(1.0, 0.5, 0.25),
        roughness_gain=0.4,
        roughness_spatial_freq=900.0,
        impact_gain=1.0,
        impact_threshold=0.4,
        contact_damping=1500.0,
    ),
    "pinky": gs.sensors.ContactAudioProperties(  # blue = ceramic
        modal_freqs=(700.0, 1600.0, 2500.0),
        modal_decays=(8.0, 12.0, 18.0),
        modal_gains=(1.0, 0.6, 0.3),
        impact_gain=1.4,
        impact_threshold=0.4,
        contact_damping=3000.0,
        accel_noise_gain=0.7,
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Shadow Hand material-tapping audio demo")
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Show visualization GUI")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-t", "--seconds", type=float, default=4.0, help="Seconds to simulate in headless mode")
    parser.add_argument("-o", "--out", type=str, default="shadow_hand_audio.wav", help="Output .wav path")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        viewer_options=gs.options.ViewerOptions(camera_pos=(0.5, -0.5, 0.5), camera_lookat=(0.0, 0.0, 0.15)),
        rigid_options=gs.options.RigidOptions(gravity=(0.0, 0.0, 0.0)),  # fixed base; pinned blocks
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane())

    # Palm-down, fixed base. (Orientation is visual; blocks are placed under the fingertips after build, so the tap
    # works regardless -- tune the euler with -v if you want a different palm pose.)
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
            pos=(0.0, 0.0, 0.22),
            quat=gu.euler_to_quat((0.0, 90.0, 0.0)),
            fixed=True,
        ),
    )

    blocks = {}
    for finger, _link, _key, color in FINGERS:
        blocks[finger] = scene.add_entity(
            gs.morphs.Box(size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), pos=(0.3, 0.0, 0.0)),
            surface=gs.surfaces.Default(color=color),
            material=gs.materials.Rigid(friction=1.0),
        )

    # ContactAudio (contact mics) on each fingertip; the struck block's link selects the material. Key -1 is a quiet
    # default for any other contact (self-collision, plane).
    props = {-1: gs.sensors.ContactAudioProperties(modal_freqs=(180.0,), modal_decays=(120.0,), modal_gains=(0.15,))}
    for finger, _link, _key, _color in FINGERS:
        props[blocks[finger].base_link_idx] = MATERIALS[finger]
    for finger, link_name, _key, _color in FINGERS:
        scene.add_sensor(
            gs.sensors.ContactAudio(
                entity_idx=robot.idx,
                link_idx_local=robot.get_link(link_name).idx_local,
                properties_dict=props,
                audio_substeps=AUDIO_SUBSTEPS,
                n_modes=N_MODES,
                draw_debug=args.vis,
            )
        )

    # Quiet actuation audio from the finger motors (a soft whirr under the taps).
    scene.add_audio_source(
        gs.audio.ActuationSource(
            entity_idx=robot.idx,
            audio_substeps=AUDIO_SUBSTEPS,
            default_properties=gs.audio.ActuationSourceProperties(
                pitch_slope=220.0,
                idle_gain=0.008,
                friction_gain=0.04,
                load_gain=0.05,
                power_gain=0.03,
                reversal_click_gain=0.04,
            ),
        )
    )
    mic = scene.add_sensor(gs.sensors.SpatialAudio(pos_offset=(0.4, -0.4, 0.35), audio_substeps=AUDIO_SUBSTEPS))
    sample_rate = int(round(AUDIO_SUBSTEPS / DT))

    scene.build()

    fingertips = [robot.get_link(link_name) for _f, link_name, _k, _c in FINGERS]
    rest_pos = {f: tensor_to_array(robot.get_link(link_name).get_pos()).reshape(3) for f, link_name, _k, _c in FINGERS}
    block_pos = {f: rest_pos[f] - np.array([0.0, 0.0, DROP]) for f in rest_pos}
    for finger, _link, _key, _color in FINGERS:
        blocks[finger].set_pos(block_pos[finger])

    robot.set_dofs_kp(np.full(robot.n_dofs, 12.0))
    robot.set_dofs_kv(np.full(robot.n_dofs, 1.0))

    tap_timer = {f: 0 for f, _l, _k, _c in FINGERS}
    is_running = True

    if args.vis:

        def stop():
            nonlocal is_running
            is_running = False

        def tap(finger):
            tap_timer[finger] = TAP_STEPS

        binds = [Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop)]
        for finger, _link, key, _color in FINGERS:
            binds.append(Keybind(f"tap_{finger}", key, KeyAction.PRESS, callback=tap, args=(finger,)))
        scene.viewer.register_keybinds(*binds)

    print("\n=== Shadow Hand material taps ===")
    print("Blocks under the fingers: index=metal(red) middle=glass(yellow) ring=wood(green) pinky=ceramic(blue)")
    print("Keys [1/2/3/4] tap index/middle/ring/pinky" if args.vis else f"Headless: scripted taps for {args.seconds}s")
    print()

    mic_blocks = []
    n_steps = int(args.seconds / DT)
    finger_names = [f for f, _l, _k, _c in FINGERS]
    try:
        step = 0
        while is_running:
            if not args.vis:  # scripted: tap each finger in turn
                f = finger_names[(step // TAP_STEPS) % 4]
                if step % TAP_STEPS == 0:
                    tap_timer[f] = TAP_STEPS // 2

            poss = []
            for finger, link_name, _key, _color in FINGERS:
                if tap_timer[finger] > 0:
                    poss.append(block_pos[finger] + np.array([0.0, 0.0, BLOCK_SIZE / 2 - 0.004]))  # press block top
                    tap_timer[finger] -= 1
                else:
                    poss.append(rest_pos[finger])
            qpos = robot.inverse_kinematics_multilink(links=fingertips, poss=poss)
            robot.control_dofs_position(qpos)

            for finger, _link, _key, _color in FINGERS:  # pin blocks (gravity is off)
                blocks[finger].set_pos(block_pos[finger])
            scene.step()
            mic_blocks.append(tensor_to_array(mic.read()).reshape(-1))

            step += 1
            if "PYTEST_VERSION" in os.environ and step >= 5:
                break
            if not args.vis and step >= n_steps:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted.")
    finally:
        if mic_blocks:
            audio = np.concatenate(mic_blocks)
            write_wav(args.out, audio, sample_rate)
            write_spectrogram(
                os.path.splitext(args.out)[0] + "_spectrogram.png",
                audio,
                sample_rate,
                title="Shadow Hand taps: metal | glass | wood | ceramic (airborne mic)",
            )
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
