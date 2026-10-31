#!/usr/bin/env python3
"""labels.json aus Ordnername + Boxposition + geprueufter Lage neu ableiten.

Warum das noetig ist: aufgenommen wurde mit PLAN_BLOCK=1 bzw. 2, der
Ordnername nennt aber die tatsaechliche Szene. In sechs von neun Collections
behauptet labels.json deshalb die falsche Boxhoehe und das falsche Bauteil -
'folien_oben_b_saeule' traegt 'HM-BS' (mitte, B-Saeule). Dazu wurde die Lage
nachtraeglich per label_lagen.py korrigiert, ohne dass slot und code
mitgezogen sind: eine Datei sagt gleichzeitig code=...-L2 und folienlage=L3.

Drei Felder gelten hier als Wahrheit, alles Abgeleitete wird daraus neu
berechnet:

    Ordnername    -> boxhoehe + bauteil   (hat immer gestimmt)
    position      -> P1..P5               (Positionen wurden eingehalten)
    folienlage    -> L1..L3               (von Hand bestaetigt)

Nicht angetastet werden Beobachtungen: retry, episode_status, recorded_at,
lage_reviewed. Die lassen sich nicht ableiten, und sie zu ueberschreiben
wuerde echte Information durch eine Annahme ersetzen. Genau daran war
store.write_labels() der falsche Weg - die Funktion setzt recorded_at und
retry neu, weil sie fuer den Aufnahmestart gedacht ist.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from record_utils.plan import schedule, store              # noqa: E402
from record_utils.label_lagen import (                     # noqa: E402
    DEFAULT_BASE, scene_from_folder, LAGEN_PRO_BAUTEIL,
)

# Beobachtet, nicht ableitbar - bleibt unberuehrt.
KEEP = ("retry", "episode_status", "recorded_at", "episode_idx",
        "collection", "episode_dir", "operator", "lage_reviewed",
        "updated_at")

# Aus dem Szenen-Index abgeleitet, aber nie beobachtet: der Plan alterniert
# sie, niemand hat protokolliert, ob das so gefahren wurde. Ein Feld, das eine
# Bedingung behauptet, die keiner geprueft hat, ist schlechter als kein Feld -
# es sieht wie eine Messung aus. Darum werden sie entfernt statt neu gesetzt.
DROP = ("handschuhe", "licht")


def block_of(scene) -> int:
    """1-basierter Szenen-Index. Scene hat kein __eq__, daher Attributvergleich."""
    for i, sc in enumerate(schedule.SCENES, 1):
        if (sc.boxhoehe, sc.bauteil) == (scene.boxhoehe, scene.bauteil):
            return i
    raise ValueError(f"Szene {scene.name} nicht im Plan")


def slot_of(scene, position: str, folienlage: str) -> int:
    """Slot-Index fuer (Szene, Boxposition, Folienlage).

    Gerechnet statt gesucht: die Schleifenordnung ist im Plan festgelegt
    (Lage innen, Position darueber).
    """
    base = schedule.block_slots(block_of(scene)).start
    p = schedule.POSITION_ORDER.index(position)
    l = schedule.LAGEN_ORDER.index(folienlage)
    return base + p * schedule.SLOTS_PER_POSITION + l


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--collection", default=None,
                    help="ohne Angabe: alle Collections")
    ap.add_argument("--apply", action="store_true",
                    help="ohne dieses Flag wird nur angezeigt")
    args = ap.parse_args()

    cols = ([args.collection] if args.collection else
            sorted(d for d in os.listdir(args.base)
                   if d.startswith("folien_")
                   and os.path.isdir(os.path.join(args.base, d))))

    n_ok = n_fix = n_skip = 0
    # attempt muss pro Slot fortlaufend sein. Stand ueberall auf 1, weil der
    # Cursor ohne 'n' nie weiterrueckte.
    attempts: dict = {}

    for col in cols:
        scene = scene_from_folder(col)
        if scene is None:
            print(f"! {col}: Szene nicht ableitbar - uebersprungen")
            continue

        root = os.path.join(args.base, col)
        eps = sorted(d for d in os.listdir(root) if d.startswith("episode_"))
        print(f"\n{col}  ({scene.title})  block {block_of(scene)}")

        for name in eps:
            ep_dir = os.path.join(root, name)
            path = os.path.join(ep_dir, "labels.json")
            if not os.path.exists(path):
                n_skip += 1
                continue
            with open(path, encoding="utf-8") as f:
                old = json.load(f)

            position = old.get("position")
            lage = old.get("folienlage")
            if position not in schedule.POSITIONEN:
                print(f"  ! {name}: position={position!r} unbekannt")
                n_skip += 1
                continue
            if lage not in LAGEN_PRO_BAUTEIL[scene.bauteil]:
                print(f"  ! {name}: folienlage={lage!r} gibt es bei "
                      f"{scene.bauteil} nicht")
                n_skip += 1
                continue

            slot = slot_of(scene, position, lage)
            fresh = schedule.labels_for(slot)
            for k in DROP:
                fresh.pop(k, None)

            key = (col, slot)
            attempts[key] = attempts.get(key, 0) + 1
            fresh["attempt"] = attempts[key]
            fresh["plan_block"] = fresh["block"]

            changed = {k: (old.get(k), v) for k, v in fresh.items()
                       if old.get(k) != v}
            stale = [k for k in DROP if k in old]
            if not changed and not stale:
                n_ok += 1
                continue

            n_fix += 1
            shown = list(changed.items())[:3]
            head = ", ".join(f"{k}: {a!r}->{b!r}" for k, (a, b) in shown)
            rest = len(changed) - len(shown)
            print(f"  {name}: {head}"
                  + (f"  (+{rest} weitere)" if rest > 0 else "")
                  + (f"  [entfernt: {', '.join(stale)}]" if stale else ""))

            if args.apply:
                # update_labels merged, loescht also nichts - die veralteten
                # Felder muessen vorher raus.
                if stale:
                    rec = store.read_labels(ep_dir)
                    for k in stale:
                        rec.data.pop(k, None)
                    store._atomic_write(rec.path, rec.data)
                store.update_labels(ep_dir, **fresh)

    print(f"\n{'GESCHRIEBEN' if args.apply else 'VORSCHAU'}: "
          f"{n_fix} zu korrigieren, {n_ok} schon korrekt, "
          f"{n_skip} uebersprungen.")
    if not args.apply and n_fix:
        print("Zum Schreiben: fix_labels.py --apply")


if __name__ == "__main__":
    main()