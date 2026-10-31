# Wrist-only, Box-Geometrie-Tokens, Cross-Embodiment via IK

Drei zusammenhängende Themen:

1. **§1** Der Test, der die offene Kernfrage schließt: *Kann die Policy die Aufgabe
   mit Wrist-only lösen, oder erkennt sie nur Palettenpositionen wieder?*
2. **§2** Box-Kontur als eigene Token-Gruppe über den freien `force`-Slot — und
   wie man erzwingt, dass die Policy sie nicht ignoriert.
3. **§3** UR10e → UR8L über IK, ohne Netzänderung.

---

## §1 Leave-One-Position-Out (LOPO)

### Warum der normale Val-Split die Frage nicht beantwortet

`cfg.task.dataset.val_ratio: 0.08` splittet über Episoden. Alle vier
Palettenpositionen liegen damit **auch im Training**. Eine Policy mit absoluter
EE-Pose im Input kann jede Val-Episode lösen, indem sie die Position
wiedererkennt und den Cluster-Mittelwert des Greifpunkts anfährt — ohne das
Bild je zu benutzen. Val-Loss ist in diesem Setup kein Beweis für visuelle
Lokalisierung. Das ist genau das Confound, das `scripts/inspect_grasp_spread.py`
im Docstring beschreibt.

### Schritt 0 — Ist die Frage überhaupt messbar?

```shell
python scripts/inspect_grasp_spread.py \
    --zarr /ros2_ws/data/zarr_datasets/foil_lifting_wrist_tesollo.zarr \
    --out /ros2_ws/data/outputs/grasp_spread.png -k 4
```

| Ergebnis | Interpretation | Konsequenz |
|---|---|---|
| `within/between ≈ 0.1` (z. B. 5 mm vs. 150 mm) | Innerhalb einer Position gibt es nichts zu lokalisieren | LOPO ist der **einzige** aussagekräftige Split. Zusätzlich: Boxpose-Randomisierung nötig, sonst ist die Frage nicht beantwortbar. |
| `within/between ≈ 1` | Echte Per-Episode-Variation vorhanden | LOPO ist sinnvoll, zusätzlich Per-Cluster-Auswertung |

### Schritt 1 — Der Split

Clustere auf der **Episoden-Startpose** (`state[episode_start, :2]`), nicht auf
dem Greifpunkt — sonst clusterst du teilweise auf die Antwort. Exakt die
`kmeans`-Logik aus `inspect_grasp_spread.py`. Dann: Cluster *k* komplett als
Val, Cluster ≠ *k* als Train. Vier Durchläufe, `k = 0..3`.

Implementierung: Episodenliste pro Cluster bestimmen, dann in
`RealDexDataset.__init__` statt des zufälligen `val_mask` eine explizite
Episoden-Maske durchreichen. Der bestehende Mechanismus (`val_ratio`, `seed`)
bleibt unangetastet, damit alte Runs vergleichbar bleiben.

### Schritt 2 — Die drei Varianten

Alle mit identischem Split, identischem Seed, identischer Epochenzahl:

| ID | Proprio-Input | Visueller Input | Was es beantwortet |
|---|---|---|---|
| **A** | absolute EE-Pose (Box-Frame) | wrist | Transferiert absolut+wrist auf eine ungesehene Position? |
| **B** | relativ (`relative_dataset.py`) | wrist | Hilft die relative Kodierung tatsächlich? |
| **C** | absolute EE-Pose | wrist **geshuffelt** | Negativkontrolle: was geht *ohne* Bildinhalt? |

Variante C braucht **kein** Neutraining — sie ist die Zeile `shuffle_wrist` in
`scripts/ablate_modalities.py`, angewandt auf den Checkpoint von A.

### Schritt 3 — Auswertung

Die einzige Zahl, die zählt, ist `xy MSE` im **Search-Window**
(`print_xy_search`, `PHASE_SEARCH`): laterale Abweichung, während die Hand noch
offen ist, einen Chunk vor dem Schließen. z und Gripper sind dort
ausgeschlossen, damit der Vorzeichenwechsel beim Greifen nicht hineinleckt.

| Beobachtung | Schlussfolgerung |
|---|---|
| **A ≪ C** | Wrist-only funktioniert. Die relative Kodierung ist dann *nicht nötig* — die absolute Variante ist einfacher und du bist fertig. |
| **A ≈ C** | A hat nur Positionen memorisiert. Kein Architektur-Trick rettet das; es fehlen Daten mit variierender Boxpose. |
| **B ≪ A ≈ C** | Die relative Kodierung ist das, was den Shortcut entfernt. Dann §2 als Ersatz für die verlorene Raumreferenz. |
| **B ≈ C** | Auch relativ löst es nicht — Kamera/Auflösung/Beleuchtung prüfen, bevor weiter an der Repräsentation gedreht wird. |

> **Aufwand:** ein Nachmittag, vorhandene Daten, vorhandene Skripte. Dieser Test
> geht **vor** jeder Architekturänderung. Er entscheidet, ob §2 überhaupt
> gebraucht wird.

---

## §2 Box-Kontur über den `force`-Slot

### Motivation

Die Top-Pointcloud trägt in diesem Prozess realistisch nur drei Informationen:
Stapelhöhe `z`, Box-/Palettenpose `(x, y, yaw)`, Zustand der obersten Lage. Das
sind ~4 Skalare plus ein Randfall — dafür 1024 Punkte durch einen PointNet zu
schicken, ist teuer, verdeckungsanfällig (der Arm steht im Bild) und
füllstandsabhängig.

Der `force`-Pfad in `ditx.py` ist vollständig verdrahtet und wird nirgends
benutzt (`maniflow_multimodal_policy.py` übergibt `force_cond` nie). Er ist der
natürliche Platz für explizite, niedrigdimensionale Geometrie.

### Repräsentation: EE-relative Wand-Tokens

Nicht die Boxpose als 6 Zahlen (das muss das Netz erst in „wie weit bis zur
Wand" umrechnen), sondern direkt die handlungsrelevante Größe — **ein Token pro
Begrenzungsfläche, ausgedrückt im aktuellen TCP-Frame**:

```python
# Vorschlag. Pro Fläche f in {wall_-x, wall_+x, wall_-y, wall_+y, floor, stack_top}
# alles im TCP-Frame des aktuellen Ticks -> embodiment-invariant, basis-frei.
token_f = [
    signed_distance,        # (1,) m, Abstand TCP -> Fläche entlang Normale
    normal_in_tcp,          # (3,) Flächennormale im TCP-Frame
    closest_point_in_tcp,   # (3,) Fußpunkt des Lots, im TCP-Frame
    onehot_type,            # (3,) [wall, floor, stack_top]
    validity,               # (1,) 0 = unbekannt/nicht vorhanden (Maskierung)
]                           # -> 11 Werte, 6 Tokens => (B, 6, 11)
```

Vier Eigenschaften, die das von einer Boxpose unterscheiden:

- **TCP-relativ** → kein Basis-Frame, numerisch identisch auf UR10e und UR8L.
- **Handlungsnah** → „3 cm bis zur linken Wand" ist direkt die Größe, die eine
  Kollision verhindert. Keine implizite Transformation nötig.
- **Mengenförmig** → Cross-Attention kann sich die *eine* nächste Wand
  herausgreifen; ein flacher Vektor erzwingt, dass alle Flächen gleichzeitig
  durch dasselbe MLP laufen.
- **Quellenagnostisch** → gespeist aus CAD + Boxpose, aus der Top-Pointcloud,
  aus einem ToF-Sensor oder aus einer Handeingabe. Die Policy sieht immer
  dasselbe Interface. `validity=0` erlaubt Degradation, wenn eine Quelle fehlt.

`stack_top` deckt dabei deine „Height-Token"-Idee ab; `floor` und die Wände
decken den Teil ab, den die Top-Kamera liefern sollte.

### Verdrahtung

Drei Stellen, alle klein:

```python
# 1) ditx.py __init__ — eigene Eingangsdimension, analog zu wrist_dim.
#    WICHTIG: der bestehende Code nutzt nn.Linear(cond_dim, n_emb), und
#    cond_dim ist in der Policy `obs_feature_dim if pc_conditioned else 1`.
#    Bei pc_conditioned=False waere force_cond_emb also nn.Linear(1, n_emb) —
#    eine stille Fehlkonfiguration. Deshalb explizit:
force_dim: int = 0                       # neuer ctor-Arg
self.force_cond_emb = nn.Linear(force_dim or cond_dim, n_emb)

# 2) forward() — bleibt wie es ist, modality_emb[2] ist schon der Geom-Slot.
#    Kein Shape-Wechsel von modality_emb => alte Checkpoints laden weiter.

# 3) maniflow_multimodal_policy.py — geom_cond bauen und durchreichen,
#    genau dort wo heute wrist_cond durchläuft: _encode_obs-Nachbarschaft,
#    compute_loss, predict_action, conditional_sample, und die
#    Flow-/Consistency-Splits (vis_cond_flow / vis_cond_ct analog slicen).
```

Dazu `geom` als Key ins Zarr (pro Frame, `(6, 11)`) und in `shape_meta.obs`.
Wichtig: **nicht** durch den `LinearNormalizer` mit `mode='limits'` jagen und
dabei die physikalische Bedeutung zerstören — Abstände in Metern sind schon
gut skaliert, Normalen und One-Hots liegen per Konstruktion in `[-1, 1]`.

### Das eigentliche Problem: wird die Information benutzt?

Dein Einwand ist der entscheidende. Eine Token-Gruppe hinzuzufügen garantiert
nichts — der Trunk kann sie wegattendieren. Vier Mechanismen, von „zwingend"
bis „nice to have":

#### (a) Informationstheoretisch notwendig machen — der einzige harte Hebel

> Wenn die Box in allen Demos am selben Ort steht, sind die Geom-Tokens eine
> **Konstante**. Eine Konstante trägt null Information und wird provably
> ignoriert. Kein Loss-Term kann das reparieren.

Also: **Boxpose über die Demos randomisieren** (±10–15 cm in x/y, ±10° yaw,
unterschiedliche Stapelhöhen). Und: wenn gleichzeitig die absolute EE-Pose im
Box-Frame im Input liegt, sind die Wandabstände daraus plus bekannter Geometrie
*ableitbar* — die Tokens sind dann redundant und werden wieder ignoriert.

Die saubere Konfiguration ist deshalb:

| | Proprio | Geom-Tokens | Folge |
|---|---|---|---|
| ✗ | absolute Pose im Box-Frame | Wände | redundant, wird ignoriert |
| ✗ | relativ | Wände, Box fix | konstant, wird ignoriert |
| **✓** | **relativ** (dein `relative_dataset.py`) | **Wände, Box randomisiert** | Tokens sind die **einzige** Quelle globaler Raumreferenz |

Damit ersetzen die Geom-Tokens exakt die Rolle, die die absolute Pose hatte —
nur ohne den Palettenpositions-Shortcut und ohne Embodiment-Abhängigkeit.

#### (b) Geometrie-only Aux-Head

Dasselbe Muster wie dein bestehender `VisionActionHead`, der Action aus
`vis_cond` **ohne** `state_cond` vorhersagt. Analog: ein Head, der aus den
Geom-Tokens allein eine Größe vorhersagt, die *nur* dort steht — z. B. die
minimale Clearance über den nächsten Chunk oder den Zeitpunkt des
Wandkontakts. Entscheidend ist die Bedingung, an der der Gelenkwinkel-Aux-Task
gescheitert ist: **das Ziel darf aus dem bevorzugten Pfad nicht ableitbar sein.**
Bei randomisierter Boxpose ist Clearance aus Bild+Proprio nicht ableitbar, aus
den Geom-Tokens trivial — genau richtig.

#### (c) Dropout auf den Geom-Tokens, nicht nur auf Proprio

`validity=0` + genullte Tokens mit ~10–20 % Wahrscheinlichkeit. Zweck ist hier
nicht Regularisierung, sondern: Degradation wird trainiert, damit ein Ausfall
der Quelle im Feld kein OOD-Zustand ist. Niedriger als
`proprio_dropout_prob: 0.8` halten — das ist eine Notmaßnahme gegen den
Proprio-Shortcut und kein Vorbild.

#### (d) Messen, nicht hoffen — zwei Probes

**Probe 1: `shuffle_geom` in `ablate_modalities.py`.** Eine Zeile:

```python
('shuffle_geom',
 lambda v, s, w, g: ...,   # Geom-Tokens über die Batch-Achse permutieren
 'box geometry content broken, stats intact  <- main geom test'),
```

> **Fallstrick:** Bei fixer Boxpose sind alle Samples im Batch identisch,
> `_shuffle` ist ein No-Op, und die Zeile meldet „ändert nichts" — unabhängig
> davon, ob die Policy die Tokens nutzt. Die Zeile ist nur bei randomisierter
> Boxpose interpretierbar. Das ist derselbe Grund wie in (a).

**Probe 2: Virtuelle Boxverschiebung (kausal, auch ohne Randomisierung).**
Verschiebe zur Inferenzzeit die Wand-Tokens um δ = 2 cm und messe, wie weit die
vorhergesagte Trajektorie mitwandert:

```
sensitivity = ||a(geom + δ) - a(geom)|| / ||δ||
```

- `≈ 0` → Tokens werden ignoriert. Zurück zu (a).
- `≈ 1` → Die Policy verschiebt ihr Ziel mit der Box. Das ist das gewünschte
  Verhalten und zugleich das, was du für Zero-Shot auf einer neuen Zellgeometrie
  brauchst.

Probe 2 ist stärker als Probe 1, weil sie eine *gerichtete* Antwort misst statt
nur „Fehler steigt". Sie ist auch der Test, den du vor dem ersten echten
UR8L-Rollout fahren willst.

---

## §3 UR10e → UR8L: Cross-Embodiment via IK

### Die Kernaussage

Die Policy bleibt **gelenkblind**. Sie arbeitet ausschließlich in
EE-Posen; die Kinematik löst der nachgeschaltete Controller. Am Netz ändert
sich **nichts** — keine Stems, kein `robot_id`, keine UR8L-Demos, bis gemessen
ist, dass es nicht reicht. Das ist dasselbe Paradigma wie UMI/Diffusion Policy,
nur ohne das Vokabular.

Längere Rohre ändern `T_base→ee(q)`. Sie ändern **nicht**, wo die Folie in der
Box liegt. Solange keine Größe im Tensor von der Armgeometrie abhängt, existiert
das Problem nicht.

### Pipeline zur Deploy-Zeit

```
Policy out: chunk-relative EE-Posen im TCP-Frame des Inferenz-Ticks
   |  from_relative(), relative_pose.py  (bereits implementiert)
   v
absolute EE-Posen im BOX-Frame
   |  T_box->base^(robot)   <- EINE kalibrierte 4x4 pro Zelle, kein gelerntes Parameter
   v
absolute EE-Posen im Basis-Frame des jeweiligen Roboters
   |  IK mit Nullraum-Regularisierung auf q_home + Limit-Abstoßung (DLS)
   v
Gelenk-Trajektorie
```

Nur der mittlere Pfeil ist roboterabhängig, und er ist eine Konstante, die man
misst. `relative_pose.py` deckt den oberen Teil schon ab und ist bewusst
shared zwischen Dataset und Deploy — genau damit die Dekodierung nicht
auseinanderdriftet.

### Was wo hingehört

| Unterschied UR10e ↔ UR8L | Richtige Ebene | Warum nicht die Policy |
|---|---|---|
| Basis-Montage, Zellgeometrie | Kalibrierung `T_box→base` | exakt lösbar, 0 Demos |
| Erreichbarkeit, Ellenbogen-Flip, `q5 ≈ 0` | IK/OSC, Nullraum auf `q_home` | Die Policy sieht Singularitäten nicht, der Solver schon |
| Statische Durchbiegung (bei langen Rohren größer) | Relative Actions absorbieren das automatisch | — |
| Regler-Lag, Trägheit, erreichbare Beschleunigung | **Hier und nur hier** sind UR8L-Demos sinnvoll | echte Dynamik, nicht Geometrie |
| Wrist-Kamera-Extrinsik `T_tcp→cam` | Mechanik/Kalibrierung | siehe unten — blocking |

### Hardware-Checkliste (vor der ersten UR8L-Aufnahme)

- [ ] `T_box→base` für beide Zellen gemessen. **Keine** Policy-Ein- oder
      -Ausgabe lebt im Basis-Frame.
- [ ] `T_tcp→cam` **identisch** auf beiden Robotern. Das ist die einzige harte
      Bedingung: 2° Verdrehung am Adapter zerstört den Wrist-only-Transfer und
      sieht dabei aus wie „zu wenig Daten". Identischer Adapter, oder Bild auf
      eine kanonische TCP-Kamera-Pose rektifizieren.
- [ ] Gleicher Greifer, gleiche TCP-Definition, gleiche `payload`/`tool_cog`
      (sonst weicht die Kraftregelung ab).
- [ ] Arbeitsraum auf dem UR8L in *einer* sauberen Konfiguration (Elbow-Up,
      weg von `q5 ≈ 0`) erreichbar.

### Offline-Dry-Run: Erreichbarkeit prüfen, ohne den UR8L anzufassen

Der billigste wertvolle Test im ganzen Dokument. Keine Hardware, keine Demos:

```
für jede Demo-Episode im Zarr:
    EE-Posen (absolut) -> T_box  -> T_base^(UR8L) -> IK(UR8L)
    protokolliere: IK-Fehlschläge, min(Limit-Margin), min(Manipulierbarkeit),
                   min|cos(q5)|, Zweigwechsel zwischen aufeinanderfolgenden Ticks
```

Das findet vor jeder Datenaufnahme heraus, ob der UR8L den Arbeitsraum
überhaupt sauber durchfährt. Ein Zweigwechsel mitten in einer Episode heißt:
IK-Nullraum nachbessern, nicht Demos sammeln.

### Was **nicht** funktioniert

> **Gelenkraum-Replay der UR10e-Demos auf dem UR8L ist ungültig.** Dieselben
> `q` ergeben bei längeren Rohren eine andere EE-Pose. Transfer funktioniert
> ausschließlich auf EE-Ebene. Umgekehrt heißt das: auf EE-Ebene sind deine
> bestehenden ~259 Episoden für den UR8L **gratis** gültig — das ist der ganze
> Punkt dieser Architektur.

### Reihenfolge

1. Extrinsik + `T_box→base` messen, Offline-Dry-Run (§3) laufen lassen.
2. Zero-Shot-Rollout auf dem UR8L mit dem UR10e-Checkpoint messen.
3. Erst bei messbarer Verschlechterung: 20–50 UR8L-Demos. Dann weißt du auch,
   *warum* — Geometrie (Kalibrierfehler, zeigt sich als konstanter Offset) oder
   Dynamik (Lag, zeigt sich als Phasenverzug unter Geschwindigkeit).
4. Ein `robot_id`-Embedding auf `time_c` (2 Zeilen, zero-init, analog zu
   `modality_emb`) ist die *letzte* Maßnahme, nicht die erste — und nur für
   Dynamik, nie für Geometrie.
