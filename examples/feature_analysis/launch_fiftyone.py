"""Launch the FiftyOne (Voxel51) app on the descriptor datasets.

Builds the datasets if they do not exist yet, then serves the app so it can be
opened from the host browser at http://localhost:5151.

Inside the container the app must bind to 0.0.0.0 — binding to localhost would
only be reachable from the container itself. `docker/docker-compose.yml` sets
`FIFTYONE_DEFAULT_APP_ADDRESS=0.0.0.0` and publishes 5151; this script also
passes the address explicitly so it works with a plain `docker run`.

Usage:

    python examples/feature_analysis/launch_fiftyone.py
    python examples/feature_analysis/launch_fiftyone.py --modes none radar_base
    python examples/feature_analysis/launch_fiftyone.py --list

Then open http://localhost:5151 on the host. Ctrl-C stops the server.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

PORT = int(os.environ.get("FIFTYONE_DEFAULT_APP_PORT", 5151))
ADDRESS = os.environ.get("FIFTYONE_DEFAULT_APP_ADDRESS", "0.0.0.0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["none", "radar_base"])
    ap.add_argument("--split", choices=["train", "val", "both"], default="val")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--list", action="store_true",
                    help="list existing datasets and exit")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild datasets even if they already exist")
    a = ap.parse_args()

    try:
        import fiftyone as fo
    except ImportError:
        raise SystemExit(
            "fiftyone is not installed in this image.\n"
            "Rebuild it:  cd docker && docker compose build tracker_multimodal_mira")

    if a.list:
        names = fo.list_datasets()
        print("  datasets:", ", ".join(names) if names else "(none)")
        return

    from analyze_features import collect, export_fiftyone

    for mode in a.modes:
        name = f"vod-features-{mode}"
        if name in fo.list_datasets() and not a.rebuild:
            print(f"  '{name}' already exists (use --rebuild to regenerate)")
            continue
        print(f"  building '{name}' ...", flush=True)
        clips_map = {
            "train": ['delft_2', 'delft_3', 'delft_4', 'delft_6', 'delft_9',
                      'delft_11', 'delft_12', 'delft_13', 'delft_19',
                      'delft_23', 'delft_24', 'delft_26', 'delft_27'],
            "val": ['delft_1', 'delft_10', 'delft_14', 'delft_22'],
        }
        clips = clips_map["train"] + clips_map["val"] if a.split == "both" \
            else clips_map[a.split]
        X, ids, frames, classes = collect(mode, clips, a.stride,
                                          a.max_frames, a.radius)
        if len(X) == 0:
            print(f"  no data for '{mode}' — skipped")
            continue
        export_fiftyone(mode, X, ids, frames, classes)

    names = [n for n in fo.list_datasets() if n.startswith("vod-features-")]
    if not names:
        raise SystemExit("no datasets to show")

    print(f"\n  serving {names[0]} on http://localhost:{PORT}")
    print(f"  datasets available: {', '.join(names)}")
    print("  (switch between them with the dataset selector in the app)")
    session = fo.launch_app(fo.load_dataset(names[0]),
                            address=ADDRESS, port=PORT, remote=True)
    session.wait()


if __name__ == "__main__":
    main()
