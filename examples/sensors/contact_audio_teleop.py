"""
Interactive ContactAudio sensor demo with keyboard teleop + WAV / MP4 export.

A "finger" pusher carries a ContactAudio sensor. Slide / tap it across three material tiles (wood, metal, glass);
each tile has different vibroacoustic properties, so impacts ring differently and sliding produces a different
velocity-pitched texture. The synthesized contact vibration is accumulated every step.

Output is chosen by the ``--out`` file extension:
  - ``.wav``  the synthesized contact audio only.
  - ``.mp4``  a rendered video of the simulation with the contact audio muxed in as its soundtrack (requires a
              camera render each step; the audio track is muxed with the bundled ffmpeg).
A ``<out>_spectrogram.png`` (waveform + log-magnitude spectrogram) is always written alongside the output.

A static airborne ``SpatialAudio`` microphone off to the side also records the contact sound radiated through the air
(distance attenuation + speed-of-sound delay), written to ``<out>_airborne.wav``. With ``--active`` the contact mic
runs in active-acoustic mode (a swept emitter excitation injected into the contacted object), written to
``<out>_active.wav`` -- its spectrogram shows the object's resonances rather than the passive scrape.

Controls (with --vis):
  [up/down/left/right]  move the finger in XY
  [j / k]               lower / raise the finger
  [space]               tap down (quick impact)
  [\\]                   reset finger position
  [esc]                 quit and write the output

Headless (no --vis) runs a scripted lower-then-slide motion so it still produces a meaningful recording.
"""

import argparse
import os
import subprocess
import tempfile
import wave

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

DT = 0.005
AUDIO_SUBSTEPS = 80  # 80 samples / 0.005 s = 16 kHz audio
N_MODES = 4

KEY_DPOS = 0.02
KEY_DPOS_Z = 0.01
FORCE_SCALE = 10.0

FINGER_SIZE = 0.05
TILE_SIZE = 0.3
TILE_HEIGHT = 0.1
FINGER_Z0 = TILE_HEIGHT + FINGER_SIZE / 2 + 0.02


def write_wav(path: str, samples: np.ndarray, sample_rate: int):
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(samples)))
    norm = samples / peak if peak > 1e-9 else samples
    pcm = np.clip(norm, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16.tobytes())
    gs.logger.info(f"Wrote {len(pcm16)} samples ({len(pcm16) / sample_rate:.2f}s @ {sample_rate} Hz) to {path}")


def write_spectrogram(path: str, samples: np.ndarray, sample_rate: int, title: str | None = None):
    """
    Save a log-magnitude spectrogram of synthesized audio (a simple STFT computed with numpy, so no scipy
    dependency). ``title`` labels the plot; defaults to a generic label. Skips gracefully if matplotlib is missing.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        gs.logger.warning("matplotlib not available, skipping spectrogram.")
        return

    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    nfft, hop = 1024, 256
    window = np.hanning(nfft)
    n_frames = max(1, 1 + (len(samples) - nfft) // hop) if len(samples) >= nfft else 1
    frames = np.zeros((n_frames, nfft))
    for i in range(n_frames):
        seg = samples[i * hop : i * hop + nfft]
        frames[i, : len(seg)] = seg
    spec = np.abs(np.fft.rfft(frames * window, axis=1))
    spec_db = 20.0 * np.log10(spec.T + 1e-6)  # (freq, time)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    times = np.arange(n_frames) * hop / sample_rate

    fig, (ax_w, ax_s) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=(1, 3), sharex=True)
    ax_w.plot(np.arange(len(samples)) / sample_rate, samples, lw=0.4, color="0.2")
    ax_w.set_ylabel("amplitude")
    ax_w.set_title(title or "Audio: waveform + spectrogram")
    vmax = spec_db.max()
    im = ax_s.pcolormesh(times, freqs, spec_db, vmin=vmax - 80.0, vmax=vmax, shading="auto", cmap="magma")
    ax_s.set_ylabel("frequency (Hz)")
    ax_s.set_xlabel("time (s)")
    ax_s.set_ylim(0, min(8000.0, sample_rate / 2))
    fig.colorbar(im, ax=ax_s, label="dB")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    gs.logger.info(f"Wrote spectrogram to {path}")


def mux_audio_video(video_path: str, wav_path: str, out_path: str):
    """
    Mux a WAV audio track onto a (silent) video file using the ffmpeg bundled with imageio_ffmpeg.
    """
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-i",
        wav_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gs.logger.info(f"Muxed audio + video to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Interactive ContactAudio sensor demo")
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Show visualization GUI")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-t", "--seconds", type=float, default=4.0, help="Seconds to simulate in headless mode")
    parser.add_argument(
        "-o", "--out", type=str, default="contact_audio.wav", help="Output path; .wav (audio) or .mp4 (video+audio)"
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate for .mp4 output")
    parser.add_argument(
        "--active",
        action="store_true",
        help="Active-acoustic mode: inject a swept excitation into the contacted object and record the modal "
        "response (Lu & Culbertson). Writes <out>_active.wav.",
    )
    args = parser.parse_args()

    out_ext = os.path.splitext(args.out)[1].lower()
    if out_ext not in (".wav", ".mp4"):
        raise SystemExit(f"--out must end in .wav or .mp4, got '{args.out}'")
    write_video = out_ext == ".mp4"

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -1.2, 1.0),
            camera_lookat=(0.0, 0.0, TILE_HEIGHT),
            camera_fov=40,
            max_FPS=60,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # Three material tiles side by side.
    tile_colors = [(0.6, 0.4, 0.2, 1.0), (0.7, 0.7, 0.75, 1.0), (0.6, 0.8, 0.9, 1.0)]
    tiles = []
    for i, color in enumerate(tile_colors):
        tile = scene.add_entity(
            gs.morphs.Box(
                size=(TILE_SIZE, TILE_SIZE, TILE_HEIGHT),
                pos=((i - 1) * (TILE_SIZE + 0.01), 0.0, TILE_HEIGHT / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=color),
            material=gs.materials.Rigid(friction=0.6),
        )
        tiles.append(tile)
    wood, metal, glass = tiles

    finger_pos_init = np.array([-(TILE_SIZE + 0.01), 0.0, FINGER_Z0], dtype=np.float32)
    finger = scene.add_entity(
        gs.morphs.Box(size=(FINGER_SIZE, FINGER_SIZE, FINGER_SIZE), pos=finger_pos_init),
        surface=gs.surfaces.Default(color=(0.2, 0.2, 0.2, 1.0)),
        material=gs.materials.Rigid(friction=0.6),
    )

    # Vibroacoustic materials keyed by the *struck* tile link. Sliding sound is the broadband noise texture colored
    # by the (contact-damped) modes; tapping pings the free ring-down. wood = low resonances + strong grainy
    # texture, modes ring out fast; metal = high resonances that ring long *after release* but are damped while
    # sliding, brighter smoother scrape; glass = very high resonances, long release ring, fine quiet scrape. Key -1
    # is a quiet default (e.g. the ground plane).
    properties_dict = {
        -1: gs.sensors.ContactAudioProperties(
            modal_freqs=(180.0,), modal_decays=(120.0,), modal_gains=(0.3,), roughness_gain=0.0, impact_gain=0.3
        ),
        wood.base_link_idx: gs.sensors.ContactAudioProperties(
            modal_freqs=(120.0, 300.0, 620.0),
            modal_decays=(55.0, 80.0, 110.0),
            modal_gains=(1.0, 0.5, 0.25),
            roughness_gain=0.5,
            roughness_spatial_freq=900.0,
            roughness_bandwidth=700.0,
            impact_gain=1.0,
            impact_threshold=0.5,
            contact_damping=1500.0,
        ),
        metal.base_link_idx: gs.sensors.ContactAudioProperties(
            modal_freqs=(1200.0, 2450.0, 3700.0),
            modal_decays=(3.0, 4.5, 6.0),
            modal_gains=(1.0, 0.7, 0.4),
            roughness_gain=1.2,
            roughness_spatial_freq=600.0,
            roughness_bandwidth=1500.0,
            impact_gain=1.6,
            impact_threshold=0.6,
            contact_damping=4000.0,
        ),
        glass.base_link_idx: gs.sensors.ContactAudioProperties(
            modal_freqs=(2000.0, 4100.0, 6300.0),
            modal_decays=(1.5, 2.5, 3.5),
            modal_gains=(1.0, 0.6, 0.3),
            roughness_gain=0.6,
            roughness_spatial_freq=450.0,
            roughness_bandwidth=1800.0,
            impact_gain=2.2,
            impact_threshold=0.7,
            contact_damping=6000.0,
        ),
    }

    # Active-acoustic mode: a swept emitter excitation injected into the contacted object's modal bank; the received
    # waveform's spectrum then reveals the object's resonances (and how the contact damps them).
    excitation = (
        gs.sensors.ExcitationSignal(kind="linear_sweep", f_lo=80.0, f_hi=7000.0, duration=0.5) if args.active else None
    )
    audio_sensor = scene.add_sensor(
        gs.sensors.ContactAudio(
            entity_idx=finger.idx,
            link_idx_local=0,
            properties_dict=properties_dict,
            audio_substeps=AUDIO_SUBSTEPS,
            n_modes=N_MODES,
            excitation=excitation,
            draw_debug=args.vis,
        )
    )
    # Airborne microphone: a static listener off to the side that hears the finger's contact sound radiated through the
    # air (distance attenuation + speed-of-sound delay). Recorded alongside the contact mic for comparison.
    mic_sensor = scene.add_sensor(
        gs.sensors.SpatialAudio(
            pos_offset=(0.8, -0.8, 0.6),
            audio_substeps=AUDIO_SUBSTEPS,
            draw_debug=args.vis,
        )
    )
    sample_rate = int(round(AUDIO_SUBSTEPS / DT))

    camera = None
    if write_video:
        camera = scene.add_camera(
            res=(960, 720),
            pos=(0.0, -1.2, 1.0),
            lookat=(0.0, 0.0, TILE_HEIGHT),
            fov=40,
            GUI=False,
        )

    scene.build()

    if camera is not None:
        camera.start_recording()

    finger.set_dofs_kp(np.full(3, FORCE_SCALE / KEY_DPOS), dofs_idx_local=slice(0, 3))
    finger.set_dofs_kv(np.full(3, 0.2 * FORCE_SCALE / KEY_DPOS), dofs_idx_local=slice(0, 3))
    finger.control_dofs_position(finger_pos_init, dofs_idx_local=slice(0, 3))

    target_pos = finger_pos_init.copy()
    is_running = True

    if args.vis:

        def stop():
            nonlocal is_running
            is_running = False

        def reset_pose():
            target_pos[:] = finger_pos_init
            finger.set_dofs_position(finger_pos_init, dofs_idx_local=slice(0, 3))

        def translate(index: int, is_negative: bool):
            target_pos[index] += (-1 if is_negative else 1) * (KEY_DPOS if index < 2 else KEY_DPOS_Z)

        def tap():
            target_pos[2] = TILE_HEIGHT + FINGER_SIZE / 2 - 0.005

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=translate, args=(1, False)),
            Keybind("move_backward", Key.DOWN, KeyAction.HOLD, callback=translate, args=(1, True)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=translate, args=(0, False)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=translate, args=(0, True)),
            Keybind("lower", Key.J, KeyAction.HOLD, callback=translate, args=(2, True)),
            Keybind("raise", Key.K, KeyAction.HOLD, callback=translate, args=(2, False)),
            Keybind("tap", Key.SPACE, KeyAction.PRESS, callback=tap),
            Keybind("reset", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_pose),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )

    print("\n=== ContactAudio demo ===")
    print(f"Audio: {AUDIO_SUBSTEPS} samples/step @ dt={DT}s  ->  {sample_rate} Hz")
    print("Tiles (left to right): wood | metal | glass")
    if args.vis:
        print("Controls: [arrows] move XY  [j/k] lower/raise  [space] tap  [\\] reset  [esc] quit")
    else:
        print(f"Running headless for {args.seconds}s with a scripted lower-then-slide motion ...")
    print()

    audio_blocks: list[np.ndarray] = []
    mic_blocks: list[np.ndarray] = []
    n_steps = int(args.seconds / DT)
    contact_z = TILE_HEIGHT + FINGER_SIZE / 2 - 0.004

    # Render a frame every `render_every` steps so the video plays back in real time (and stays in sync with the
    # audio). effective_fps is the real-time frame rate handed to the encoder.
    render_every = max(1, round((1.0 / DT) / args.fps)) if write_video else 0
    effective_fps = (1.0 / DT) / render_every if write_video else args.fps

    try:
        step = 0
        while is_running:
            if not args.vis:
                # Scripted: press down for the first ~0.4s, then slide across all three tiles in +x.
                if step < int(0.4 / DT):
                    target_pos[2] = contact_z
                else:
                    target_pos[0] = finger_pos_init[0] + 2.2 * (TILE_SIZE + 0.01) * (step / n_steps)
                    target_pos[2] = contact_z

            finger.control_dofs_position(target_pos, dofs_idx_local=slice(0, 3))
            scene.step()

            if args.vis:
                cur = tensor_to_array(finger.get_pos())
                target_pos[:2] = np.clip(target_pos[:2] - cur[:2], -KEY_DPOS, KEY_DPOS) + cur[:2]

            audio_blocks.append(tensor_to_array(audio_sensor.read()).reshape(-1))
            mic_blocks.append(tensor_to_array(mic_sensor.read()).reshape(-1))
            if camera is not None and step % render_every == 0:
                camera.render()

            step += 1
            if "PYTEST_VERSION" in os.environ and step >= 5:
                break
            if not args.vis and step >= n_steps:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted.")
    finally:
        if audio_blocks:
            audio = np.concatenate(audio_blocks)
            if write_video:
                with tempfile.TemporaryDirectory() as tmp:
                    wav_tmp = os.path.join(tmp, "audio.wav")
                    video_tmp = os.path.join(tmp, "video.mp4")
                    write_wav(wav_tmp, audio, sample_rate)
                    camera.stop_recording(save_to_filename=video_tmp, fps=effective_fps)
                    mux_audio_video(video_tmp, wav_tmp, args.out)
            else:
                write_wav(args.out, audio, sample_rate)
            write_spectrogram(
                os.path.splitext(args.out)[0] + "_spectrogram.png",
                audio,
                sample_rate,
                title="ContactAudio: waveform + spectrogram (wood | metal | glass)",
            )

            base = os.path.splitext(args.out)[0]
            if args.active:
                # The contact-mic recording is the active-acoustic response; save it under a clear name too.
                write_wav(base + "_active.wav", audio, sample_rate)
            if mic_blocks:
                mic_audio = np.concatenate(mic_blocks)
                write_wav(base + "_airborne.wav", mic_audio, sample_rate)
                write_spectrogram(
                    base + "_airborne_spectrogram.png",
                    mic_audio,
                    sample_rate,
                    title="SpatialAudio (airborne mic): waveform + spectrogram",
                )
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
