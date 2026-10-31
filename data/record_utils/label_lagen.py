#!/usr/bin/env python3
"""Folienlage pro Episode nachlabeln - klicken statt tippen.

Die Lage ist die EINE Bedingung, die der Recorder nicht ableiten konnte: der
Cursor rueckte nur bei 'n' weiter, und wer das nicht drueckt, bekommt fuer
jede Episode einer Boxposition dieselbe Lage ins labels.json geschrieben. Das
Feld behauptet dann etwas Falsches, und zwar still.

Zwei Schritte, weil pyzed nur im zed_container steckt und die GUI ein Display
braucht:

    # 1. letzter Frame je Episode als PNG (im zed_container)
    python3 label_lagen.py thumbs --collection folien_mitte_a_saeule

    # 2. durchklicken (auf dem Host)
    python3 label_lagen.py review --collection folien_mitte_a_saeule

Der LETZTE Frame wird gezeigt, nicht der erste: am Ende ist die Folie weg und
man sieht, welche Ebene freigelegt wurde. Am Anfang sieht jede Lage gleich aus.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from record_utils.plan import schedule, store  # noqa: E402

EPISODES_SUBDIR = os.environ.get("EPISODES_SUBDIR", "external")

# Ein und dasselbe Verzeichnis (/mnt/data/data auf dem Host) ist in den
# Containern unterschiedlich gemountet: zed_container sieht es unter
# /root/ros2_ws/data, ros2_dev unter /ros2_ws/data. 'thumbs' muss in
# zed_container laufen (nur dort gibt es pyzed), 'review' in ros2_dev (dort
# liegen matplotlib und Tk) - derselbe Aufruf muss also in beiden
# funktionieren. Mit einem festen Default meldet der zweite Schritt sonst
# "keine Thumbnails", obwohl der erste sie gerade erzeugt hat.
_WORKDIR_CANDIDATES = (
    os.environ.get("WORKDIR"),
    "/ros2_ws/data/",            # ros2_dev
    "/root/ros2_ws/data/",       # zed_container
)


def _default_base():
    for workdir in _WORKDIR_CANDIDATES:
        if not workdir:
            continue
        path = os.path.join(workdir, "episodes", EPISODES_SUBDIR)
        if os.path.isdir(path):
            return path
    # Host: der Mount selbst, ohne episodes/<subdir>-Zwischenebene.
    if os.path.isdir("/mnt/data/data"):
        return "/mnt/data/data"
    return os.path.join(_WORKDIR_CANDIDATES[1], "episodes", EPISODES_SUBDIR)


DEFAULT_BASE = _default_base()

THUMB_NAME = "review_last_frame.jpg"

# Die Heckleuchte hat nur zwei Lagen: die Bodenfolie ueber der Pappe sieht bei
# allen drei Bauteilen gleich aus und wurde bewusst weggelassen, um sie nicht
# zu ueberrepraesentieren. L3 darf hier also gar nicht angeboten werden - eine
# Lage anzubieten, die es nicht gab, erzeugt genau die falschen Labels, die
# dieses Tool beseitigen soll.
LAGEN_PRO_BAUTEIL = {
    "a_saeule":    ["L1", "L2", "L3"],
    "b_saeule":    ["L1", "L2", "L3"],
    "heckleuchte": ["L1", "L2"],
}


def scene_from_folder(name):
    """'folien_oben_a_saeule' -> Scene. Der Ordnername ist hier die Wahrheit.

    Nicht aus labels.json gelesen: dort steht der Block, mit dem aufgenommen
    wurde, und der war in sechs von neun Collections falsch gesetzt. Der
    Ordnername hat in jedem Fall gestimmt.
    """
    key = name.lower()
    if key.startswith("folien_"):
        key = key[len("folien_"):]
    # 'sauele' ist ein Tippfehler in drei Ordnernamen, der auf der Platte
    # steht und nicht umbenannt werden soll, nur weil er haesslich ist.
    key = key.replace("sauele", "saeule")
    for hoehe in schedule.HOEHEN_ORDER:
        if key.startswith(hoehe + "_"):
            bauteil = key[len(hoehe) + 1:]
            if bauteil in schedule.BAUTEILE:
                return schedule.Scene(hoehe, bauteil)
    return None


# ── Schritt 1: Thumbnails ───────────────────────────────────────────────────
def cmd_thumbs(args):
    import pyzed.sl as sl
    import cv2

    cols = _collections(args.base, args.collection)
    pairs = [(c, d) for c in cols
             for d in sorted(os.listdir(os.path.join(args.base, c)))
             if d.startswith("episode_")]
    done = skipped = failed = 0
    broken = []

    for i, (col, name) in enumerate(pairs, 1):
        root = os.path.join(args.base, col)
        ep_dir = os.path.join(root, name)
        out = os.path.join(ep_dir, THUMB_NAME)
        if os.path.exists(out) and not args.force:
            skipped += 1
            continue
        svo = os.path.join(ep_dir, "recording.svo2")
        if not os.path.exists(svo):
            failed += 1
            broken.append(f"{col}/{name} (keine SVO)")
            continue

        cam = sl.Camera()
        init = sl.InitParameters()
        init.set_from_svo_file(svo)
        init.svo_real_time_mode = False
        # DEPTH aus: fuer ein Vorschaubild ist Tiefe unnoetig und kostet die
        # meiste Zeit pro Episode.
        init.depth_mode = sl.DEPTH_MODE.NONE
        if cam.open(init) != sl.ERROR_CODE.SUCCESS:
            failed += 1
            broken.append(f"{col}/{name} (SVO nicht lesbar)")
            continue

        n = cam.get_svo_number_of_frames()
        img = sl.Mat()
        ok = False
        # Mehrere Positionen von hinten nach vorne: der letzte Frame ist
        # manchmal halb geschrieben, wenn die Aufnahme mitten im Frame
        # gestoppt wurde. Genau daran war beim ersten Lauf eine Episode
        # gescheitert.
        for back in (5, 15, 40, 1):
            cam.set_svo_position(max(0, n - back))
            if cam.grab() != sl.ERROR_CODE.SUCCESS:
                continue
            cam.retrieve_image(img, sl.VIEW.LEFT)
            frame = img.get_data()[:, :, :3]
            h = 540
            w = int(frame.shape[1] * h / frame.shape[0])
            cv2.imwrite(out, cv2.resize(frame, (w, h)),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            ok = True
            break
        cam.close()
        done += ok
        if not ok:
            failed += 1
            broken.append(f"{col}/{name} (kein lesbarer Frame)")
        print(f"  [{i}/{len(pairs)}] {col}/{name} "
              f"{'ok' if ok else 'FEHLER'}")

    print(f"\n{done} erzeugt, {skipped} schon da, {failed} fehlgeschlagen.")
    for b in broken:
        print(f"  ! {b}")


# ── Schritt 2: GUI ──────────────────────────────────────────────────────────
def _collections(base, only=None):
    """Alle Plan-Collections in Plan-Reihenfolge, oder nur die angegebene."""
    if only:
        return [only]
    have = {d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))}
    out = []
    # Plan-Reihenfolge statt alphabetisch: so entspricht das Durchklicken der
    # Reihenfolge, in der aufgenommen wurde.
    for sc in schedule.SCENES:
        for name in have:
            s = scene_from_folder(name)
            if s and (s.boxhoehe, s.bauteil) == (sc.boxhoehe, sc.bauteil):
                out.append(name)
    return out


def _episodes(base, collection):
    root = os.path.join(base, collection)
    if not os.path.isdir(root):
        return []
    return [os.path.join(root, d) for d in sorted(os.listdir(root))
            if d.startswith("episode_")
            and os.path.exists(os.path.join(root, d, THUMB_NAME))]


def cmd_review(args):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    cols = _collections(args.base, args.collection)

    # Alle Episoden aller Collections in EINER Liste: ein Fenster, das von
    # vorne bis hinten durchlaeuft. Pro Collection neu zu starten bedeutet bei
    # neun Ordnern neun Mal Fenster auf, Pfad tippen, Fenster zu - und die
    # Lagen sind ueber alle Collections dieselbe Frage.
    items = []
    for col in cols:
        scene = scene_from_folder(col)
        if scene is None:
            print(f"  ! {col}: Szene nicht ableitbar, uebersprungen")
            continue
        for ep in _episodes(args.base, col):
            items.append((ep, col, scene, LAGEN_PRO_BAUTEIL[scene.bauteil]))

    if not items:
        sys.exit(f"Keine Thumbnails unter {args.base}. Erst 'thumbs' laufen "
                 f"lassen (im zed_container, wegen pyzed).")

    def current(ep_dir):
        rec = store.read_labels(ep_dir)
        return (rec.data.get("folienlage") if rec else None,
                bool(rec.data.get("lage_reviewed")) if rec else False)

    # Dort anfangen, wo noch nichts bestaetigt ist: nach einer Pause will man
    # nicht erst durch die bereits erledigten Episoden blaettern.
    start = next((i for i, it in enumerate(items) if not current(it[0])[1]), 0)
    state = {"idx": start, "history": []}

    done_total = sum(1 for it in items if current(it[0])[1])
    print(f"\n{len(items)} Episoden in {len(cols)} Collections, "
          f"{done_total} schon bestaetigt.")
    print("  1/2/3  Lage setzen (springt weiter)")
    print("  <- ->  blaettern        n/b  naechste/vorige Collection")
    print("  u      zurueck         q    Ende (Fortschritt ist gespeichert)\n")

    fig, ax = plt.subplots(figsize=(13, 8))

    def draw():
        ax.clear()
        ep, col, scene, lagen = items[state["idx"]]
        ax.imshow(mpimg.imread(os.path.join(ep, THUMB_NAME)))
        ax.axis("off")
        lage, reviewed = current(ep)
        menu = "   ".join(f"[{i+1}] {L} {schedule.FOLIENLAGEN[L]}"
                          for i, L in enumerate(lagen))
        n_col = sum(1 for it in items if it[1] == col)
        i_col = sum(1 for it in items[:state["idx"] + 1] if it[1] == col)
        done = sum(1 for it in items if current(it[0])[1])
        fig.suptitle(
            f"[{state['idx']+1}/{len(items)}]  {done} bestaetigt   "
            f"|   {scene.title}  ({i_col}/{n_col})\n"
            f"{os.path.basename(ep)}   Lage: {lage or '?'}  "
            f"{'✔ bestaetigt' if reviewed else '· ungeprueft'}\n{menu}",
            fontsize=11, fontweight="bold",
            color="green" if reviewed else "darkorange")
        fig.canvas.draw_idle()

    def set_lage(lage):
        ep, _, _, _ = items[state["idx"]]
        prev, _ = current(ep)
        state["history"].append((state["idx"], prev))
        store.update_labels(
            ep,
            folienlage=lage,
            folienlage_label=schedule.FOLIENLAGEN[lage],
            # Markiert, dass ein Mensch das Bild gesehen hat. Ohne die Marke
            # ist ein korrektes Label nicht von einem zufaellig richtigen
            # abgeleiteten zu unterscheiden.
            lage_reviewed=True,
        )
        print(f"  {os.path.basename(ep)} -> {lage}")
        if state["idx"] < len(items) - 1:
            state["idx"] += 1
        draw()

    def jump_collection(step):
        cur = items[state["idx"]][1]
        order = [c for c in dict.fromkeys(it[1] for it in items)]
        i = order.index(cur) + step
        if 0 <= i < len(order):
            state["idx"] = next(j for j, it in enumerate(items)
                                if it[1] == order[i])
            draw()

    def on_key(event):
        lagen = items[state["idx"]][3]
        if event.key in [str(i + 1) for i in range(len(lagen))]:
            set_lage(lagen[int(event.key) - 1])
        elif event.key == "right":
            state["idx"] = min(len(items) - 1, state["idx"] + 1); draw()
        elif event.key == "left":
            state["idx"] = max(0, state["idx"] - 1); draw()
        elif event.key == "n":
            jump_collection(+1)
        elif event.key == "b":
            jump_collection(-1)
        elif event.key == "u" and state["history"]:
            idx, prev = state["history"].pop()
            store.update_labels(items[idx][0], folienlage=prev,
                                lage_reviewed=False)
            state["idx"] = idx
            print(f"  zurueck: {os.path.basename(items[idx][0])} -> {prev}")
            draw()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.tight_layout()
    plt.show()

    print("\nFortschritt je Collection:")
    for col in cols:
        eps = [it[0] for it in items if it[1] == col]
        if eps:
            n = sum(1 for e in eps if current(e)[1])
            flag = "" if n == len(eps) else "   <- offen"
            print(f"  {col:<34} {n:>3}/{len(eps):<3}{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help=f"Episoden-Wurzel (erkannt: {DEFAULT_BASE})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("thumbs", help="letzten Frame je Episode extrahieren")
    t.add_argument("--collection", default=None,
                   help="ohne Angabe: alle Collections")
    t.add_argument("--force", action="store_true")
    t.set_defaults(func=cmd_thumbs)

    r = sub.add_parser("review", help="durchklicken und Lage setzen")
    r.add_argument("--collection", default=None,
                   help="ohne Angabe: alle Collections hintereinander")
    r.set_defaults(func=cmd_review)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()