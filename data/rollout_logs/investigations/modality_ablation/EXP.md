# Experiment-Masterlog: Single-Variable Ablationen & Modell-Evolution

Dieses Dokument dient als zentrales Hauptfile zur Dokumentation aller Experimente, von den ersten Baselines über die Modalitäts-Ablation bis hin zu den schrittweisen, streng isolierten Folgeexperimenten (insbesondere der relativen Aktions- und Zustandsrepräsentation basierend auf *mimic-one* / UMI).

**Grundsatz:** Pro Experiment wird **exakt eine isolierte Variable** geändert, um Kausalitäten zweifelsfrei zuzuordnen.

---

## 1. Vergangene Experimente: Chronologischer Verlauf

### Phase 0: Initial-Baselines & Erste Real-Robot Rollouts (Januar 2026)
Untersuchung von absoluter vs. relativer Aktionsrepräsentation auf dem echten Roboter (23 relative vs. 19 absolute Rollouts, dokumentiert in [`findings.md`](file:///home/mm/dev/il-visual-tactile/data/rollout_logs/foil_lifting_relative_e40/findings.md)).

| Checkpoint | Repräsentation | Modalitäten | Rollout-Erfolg | Kernbefund am Rig |
| :--- | :--- | :--- | :--- | :--- |
| `old_abs_0112` | Absolut (15D) | PC (xyz) + Wrist-RGB | 21.1 % (4/19) | Stark sensitiv auf Startposition (<30 mm Startfehler $\to$ 80 % Erfolg, sonst Abbruch). Unterreagiert bei Höhenversatz (Gain 0.63). |
| `old_rel_0112` | Relativ (17D) | PC (xyz+rgb) + Wrist-RGB | 30.4 % (7/23) | Immun gegen Startposen-Versatz. Greifen sauberer (+0.4 mm Aperture Gap). **Überreagiert massiv auf Z-Höhe (Gain 1.50)**. Keine globale Verankerung. |

*Haupterkenntnis Phase 0:* Rückintegration (`test_deploy_relative_roundtrip.py`) mathematisch fehlerfrei ($< 10^{-5}\,\text{m}$). Die Z-Überreaktion und breite Streuung sind policy-immanent.

---

### Phase 1: Modality Ablation Matrix & Diagnostik (8 Modelle, 80 Epochen)
Systematischer Vergleich aller Sensormodalitäten (PC, Wrist, Front) für Absolute und Relative Targets auf identischem Datensatz ([`ablation_and_rollout_analysis.md`](file:///home/mm/dev/il-visual-tactile/data/rollout_logs/investigations/modality_ablation/ablation_and_rollout_analysis.md)).

| Modell-ID | Bedingung | Targets | Val MSE | Spread ($K=8$) mm / deg | Seam ($\Delta t=8$) mm / deg | Dominante Modalität (Ablation) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0015` | PC + Wrist | Absolut | 0.0898 | 2.10 mm / 0.16° | 1.80 mm / 0.40° | Proprio (1.73x), Wrist (1.30x), PC (1.56x) |
| `0016` | Wrist-only | Absolut | 0.1017 | 10.93 mm / 0.77° | 6.84 mm / 1.13° | Proprio (1.81x), Wrist (1.61x) |
| `0017` | Wrist + Front | Absolut | 0.0963 | 11.23 mm / 0.78° | 7.74 mm / 1.09° | Proprio (1.83x), Wrist (1.63x), Front (1.00x) |
| `0019` | PC + Wrist | Relativ | 0.0375 | 1.60 mm / 0.14° | 0.79 mm / 0.20° | Wrist (1.44x), PC (1.39x), State arm-tot (1.00x) |
| `0020` | Wrist-only | Relativ | 0.0401 | 5.37 mm / 0.39° | 2.39 mm / 0.48° | Wrist (1.98x), State arm-tot (1.00x) |
| `0021` | Wrist + Front | Relativ | 0.0402 | 4.88 mm / 0.35° | 2.29 mm / 0.45° | Wrist (1.93x), Front (1.00x), State arm-tot (1.00x) |

*Kernbefunde Phase 1:*
1. **Front-RGB ist redundant:** Bietet gegenüber Wrist-only keinen messbaren Informationsgewinn (`1.00x` Einfluss, identischer Val-Loss).
2. **PC-Dissektion:** Die relative Policy nutzt die Punktwolke fast ausschließlich für die vertikale Z-Achse (`1.42x` Fehler bei Shuffle) und kaum für XY (`1.12x`).
3. **State-History Bug entdeckt:** In `_encode_obs` schnitt `state_feat[:, -1:, :]` die Vergangenheits-Posen ab. Da bei relativen Modellen der Schritt $t$ per Definition $(0, I)$ ist, war die Arm-Propriozeption im DiTX wirkungslos (`1.00x`).

---

## 2. Laufende & Geplante Experiment-Matrix

| ID | Experiment | Status | Basis-Modelle | Geänderte Variable | Ziel & Hypothese |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | `state_history_fix` | **Wird trainiert** | `0019`, `0020`, `0021` | `state_history=true` in `_encode_obs` (alle $T_o=2$ Zeitschritte im DiTX) | Liefert $P_{t-1}^{\text{rel}}$ Geschwindigkeit/Richtung; senkt Seam-Jump & Multi-Sample Spread. |
| **Exp 2** | `wrist_plus_scalar_z` | Bereit (Configs & Smoke-Test fertig) | `abl_rel_wrist_height_*`, `abl_abs_wrist_height_*` | `agent_pos` + 1D Höhen-Skalar ($\Delta z$), kein PointNet | Reicht lokale Höheninformation am Greifer ohne Top-Kamera-Punktwolke für Z-Stabilität? |
| **Exp 3** | `temporal_window` | Bereit (Configs erstellt) | Exp 2 (`wrist + delta_z`) | Zeitfenster von $33\,\text{ms}$ auf $\approx 100\,\text{ms}$ ($n_{\text{obs\_steps}}=4$) auf Basis von Exp 2 | Reduziert Phasenambiguität am Übergang `approach -> search -> transition`. |

---

## 3. Detail-Spezifikation der Folgeexperimente

### Experiment 1: `state_history_fix` (Aktuell in Ausführung)

### Hintergrund & Befund
In der Ausgangsimplementierung ([`maniflow_multimodal_policy.py`](file:///home/mm/dev/il-visual-tactile/third_party/ManiFlow_Policy/ManiFlow/maniflow/policy/maniflow_multimodal_policy.py#L340-L378)) wurde `state_feat` der Form $(B, T_o, 32)$ durch `state_feat[:, -1:, :]` auf den letzten Zeitschritt beschnitten. 
Da bei relativen Policies der letzte Schritt $P_t^{\text{rel}}$ per Definition die Einheits-Pose $(0, I)$ ist, wurde der vergangene relative Bewegungsvektor $P_{t-1}^{\text{rel}}$ verworfen. Der DiTX-Backbone sah für den Arm ausschließlich die konstante Identität.

### Einzige Änderung
- Einführung von `policy.state_history: true`.
- Statt `state_feat[:, -1:, :]` wird die gesamte State-Historie $(B, 1, T_o \cdot D_{\text{state}}) \to (B, \text{horizon}, 64)$ konkateniert und in den DiTX-Eingang eingespeist.
- Alle alten Checkpoints bleiben über `policy.state_history=false` (Default) 100 % abwärtskompatibel.

### Trainings-Konfigurationen (80 Epochen, Seed 0)
1. **0022_rel_wrist_pc_statehist**:
   - Task: `ablation/abl_rel_wrist_pc` (`point_cloud` 1024x6 + `wrist_rgb`)
   - Override: `policy.state_history=true`
2. **0023_rel_wrist_only_statehist**:
   - Task: `ablation/abl_rel_wrist` (`wrist_rgb` only, kein PointNet)
   - Override: `policy.state_history=true`
3. **0024_rel_wrist_front_statehist**:
   - Task: `ablation/abl_rel_wrist_front` (`wrist_rgb` + `front_rgb`, kein PointNet)
   - Override: `policy.state_history=true`

### Start-Skript
Ausgeführt im `ros2_dev`-Container via:
```bash
bash /ros2_ws/third_party/ManiFlow_Policy/scripts/run_exp1_relative_statehist.sh
```

### Rollback- & Lösch-Anweisung
Sollte Experiment 1 keine statistisch signifikante Verbesserung bei Seam-Disagreement ($\Delta t = 8$) oder Modality-Ablation (`shuffle_state`) zeigen:
```bash
# Löschen der 3 Exp 1 Checkpoints und Hydra-Outputs:
rm -rf /home/mm/dev/il-visual-tactile/data/outputs/ablation/*0022*
rm -rf /home/mm/dev/il-visual-tactile/data/outputs/ablation/*0023*
rm -rf /home/mm/dev/il-visual-tactile/data/outputs/ablation/*0024*
```

---

### Experiment 2: `wrist_plus_scalar_z` (Startklar vorbereitet)

### Motivation & Tradeoff
Die Top-Kamera liefert eine globale Punktwolke (1024 Punkte), die bei veränderten Palettenpositionen oder Kameramontagefehlern kalibrierempfindlich ist. Die Kanal-Dissektion zeigte, dass die relative Policy die Punktwolke fast ausschließlich für die Z-Koordinate ($1.42\times$) und kaum für XY ($1.12\times$) nutzt. 
**Kernfrage:** Kann man auf die Top-Kamera und das rechenintensive PointNet vollständig verzichten, indem man der Wrist-Only-Policy einen kompakten lokalen Höhen-Skalar übergibt?

### Isolierte Variablenänderung
- Basis: `wrist_only` (kein PointNet, keine Top-Kamera-X/Y).
- Erweiterung von `agent_pos` um 1 Skalar via [`height_extractor.py`](file:///home/mm/dev/il-visual-tactile/third_party/ManiFlow_Policy/ManiFlow/maniflow/dataset/height_extractor.py) (17D $\to$ 18D relativ bzw. 15D $\to$ 16D absolut):
  - **Variante 2a (Relativer Oberflächenabstand $\Delta z_{\text{surface}}$):**
    $$\Delta z = z_{\text{ee}} - z_{\text{foil}}$$
    *Eigenschaft:* Strikt invariant gegen globale Palettenhöhenversätze (simuliert einen 1D-Time-of-Flight / Laser-Distanzsensor an der Hand).
  - **Variante 2b (Absolute Folienhöhe $z_{\text{foil}}$):**
    Direkter Vergleichswert zur Prüfung, ob die Policy absolute oder relative Höhen bevorzugt.

### Trainings-Konfigurationen:
- Relativ Delta: `ablation/abl_rel_wrist_height_delta`
- Relativ Abs: `ablation/abl_rel_wrist_height_abs`
- Absolut Delta: `ablation/abl_abs_wrist_height_delta`
- Absolut Abs: `ablation/abl_abs_wrist_height_abs`

---

### Experiment 3: `temporal_window_scaling` (Startklar vorbereitet)

### Motivation
Im Vorbild *mimic-one* läuft die Regelung bei $15\,\text{Hz}$ ($\Delta t = 66.7\,\text{ms}$ pro Schritt), wodurch $H_o = 2$ Zeitschritte einen zeitlichen Rückblick von $\approx 67\text{–}133\,\text{ms}$ bieten.
In unserem Setup mit $30\,\text{Hz}$ decken $n_{\text{obs\_steps}} = 2$ bei Stride 1 lediglich $33\,\text{ms}$ ab. Dies ist extrem kurz, um Geschwindigkeiten oder Phasenwechsel (`approach -> search -> hold`) rauscharm zu schätzen.

### Isolierte Variablenänderung
- Basis: **`wrist_plus_scalar_z` aus Experiment 2** (keine Pointcloud).
- Skalierung des Beobachtungshorizonts auf $n_{\text{obs\_steps}} = 4$:
  - Rückblick von $4 \times 33\,\text{ms} = 133\,\text{ms}$.
  - $4 \times 192 = 768$ visuelle Wrist-Tokens in DiTX.
  - 4 Zeitschritte State-Historie im DiTX-Input ($4 \times 32 = 128\,\text{dim}$).

### Trainings-Konfigurationen:
- Relativ: `ablation/abl_rel_wrist_height_delta_to4`
- Absolut: `ablation/abl_abs_wrist_height_delta_to4`

### Hypothesen & Prüfkriterien
1. Reduziert ein $133\,\text{ms}$-Zeitfenster die Chunk-Seam-Uneinigkeit ($\Delta t = 8$) an den Phasenübergängen?
2. Werden die Prädiktionen in der `search`-Phase stabiler gegen Sensorrauschen der Wrist-Kamera?

