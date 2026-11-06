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

## 2. Experiment-Matrix (Exp 1–3 Abgeschlossen)

> [!IMPORTANT]
> **Datensatz-Hinweis (`wrist_tesollo.zarr` vs. `box_recorded_action.zarr`):**
> Alle 9 Runs aus Exp 1–3 (`0022`–`0030`) wurden auf `foil_lifting_wrist_tesollo.zarr` (252 Episoden, Target $a_t = s_{t+1}$) trainiert – exakt identisch zu den Phase-0-Baselines `old_abs_0112` und `old_rel_0112`. Die Phase-1-Runs (`0015`–`0021`) liefen hingegen auf `foil_lifting_box_recorded_action.zarr` (306 Episoden, unglättetes Kommandotarget $a_t^{\text{cmd}}$ mit $\max|s_{t+1}-a_t|=22.84$). Dadurch sind `0022`–`0030` untereinander sowie direkt gegen `old_abs_0112` / `old_rel_0112` zu 100 % kontrolliert vergleichbar; gegenüber `0015`–`0021` sollten primär normierte Faktoren (`x_full`, `rel_spread_xy`) verglichen werden.

| ID | Experiment | Status | Modelle | Geänderte Variable | Hauptergebnis (Prüfung der Hypothese) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | `state_history_fix` | **Abgeschlossen** | `0022`, `0023`, `0024` | `state_history=true` ($T_o=2$, $33\,\text{ms}$) | **Teilbestätigt:** 3D-Spread sinkt massiv (`0022`: **5.82 mm** vs. `old_rel_0112`: **10.63 mm**, $-45\,\%$; Seam-p95: **16.85 mm** vs. **23.45 mm**). Aber: isolierter $t-1$-Arm-Pose-Einfluss (`shuffle_state_hist`) bei $33\,\text{ms}$ nur `1.04x–1.08x` (da $2\,\text{mm}$ Noise $\approx$ $1\text{–}3\,\text{mm}$ Schrittweite pro Frame). |
| **Exp 2** | `wrist_plus_scalar_z` | **Abgeschlossen** | `0025`, `0026` (rel)<br>`0027`, `0028` (abs) | `agent_pos` + 1D Höhen-Skalar ($\Delta z$ vs. $z_{\text{foil}}$), kein PointNet | **Bestätigt für $\Delta z$, widerlegt für $z_{\text{foil}}$:** Relatives $\Delta z$ (`0025`) wird stark genutzt (`shuffle_height = 1.30x`, Z-Achse `1.59x`, Approach `1.92x`) und senkt Spread (`7.41 mm` vs. `8.26 mm` in `0023`). Globales $z_{\text{foil}}$ (`0026`, `0028`) wird komplett ignoriert (`1.00x` / `0.97x`)! |
| **Exp 3** | `temporal_window` | **Abgeschlossen** | `0029` (rel, $T_o=4$)<br>`0030` (abs, $T_o=4$) | Beobachtungsfenster $33\,\text{ms} \to 133\,\text{ms}$ ($n_{\text{obs\_steps}}=4$) auf Basis von Exp 2 ($\Delta z$) | **Voll bestätigt:** In `0029` steigt `shuffle_state_hist` von `1.13x` auf **`1.29x`** (Z: `1.40x`, `xy_search`: `1.35x`, Approach: `1.70x`, Hold: `1.95x`). **Niedrigster Seam-Jump aller Modelle (`8.13 mm` Median, Z-Seam `3.35 mm`)** und bester relativer `xy_search`-MSE (`0.00699`) ganz ohne Top-Kamera/PointNet! |

### Gesamtübersicht der Evaluation (Stratified Val: 256 Episoden-Fenster, $K=8$ Spread, $\Delta t=8$ Seam)

| Modell | Repr. | Modalitäten | Best Ckpt Val MSE | Stratified Full MSE | `xy_search` MSE | Spread 3D / XY / Z (mm) | `rel_spread_xy` (Search) | Seam 3D / XY / Z Med (mm) | Seam p95 (mm) | Key Ablation (`x_full`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `old_abs_0112` | Abs ($T_o=2$) | PC + Wrist | 0.0155 | 0.02353 | 0.00379 | 9.83 / 7.07 / 6.53 | 0.375 (0.343) | 13.59 / 9.96 / 6.05 | 26.65 | State `2.97x`, PC `1.70x` (XY `11.6x`), Wrist `1.10x` |
| `old_rel_0112` | Rel ($T_o=2$) | PC + Wrist | 0.0221 | 0.02803 | 0.01037 | 10.63 / 7.45 / 7.26 | 0.190 (0.177) | 11.11 / 7.54 / 6.56 | 23.45 | PC `1.69x`, State `1.49x`, Wrist `1.39x` |
| **Exp 1 (`0022`)** | Rel ($T_o=2$) | PC + Wrist + `statehist` | 0.0238 | 0.02852 | 0.00801 | **5.82 / 4.03 / 4.02** | **0.103 (0.094)** | 10.28 / 7.56 / 4.75 | **16.85** | State `1.54x` (Hist `1.06x`, Curr `1.31x`), PC `1.51x`, Wrist `1.49x` |
| **Exp 1 (`0023`)** | Rel ($T_o=2$) | Wrist-only + `statehist` | 0.0240 | 0.03173 | 0.00809 | 8.26 / 5.64 / 5.72 | 0.144 (0.134) | 11.01 / 8.03 / 6.38 | 20.87 | Wrist `1.97x`, State `1.53x` (Hist `1.08x`, Curr `1.33x`) |
| **Exp 1 (`0024`)** | Rel ($T_o=2$) | Wrist + Front + `statehist` | **0.0197** | **0.02484** | 0.00930 | 6.95 / 4.79 / 4.80 | 0.123 (0.115) | 10.57 / 8.51 / 4.51 | 19.36 | Wrist `2.05x`, State `1.55x`, **Front `1.42x`** (Hist `1.04x`) |
| **Exp 2 (`0025`)** | Rel ($T_o=2$) | Wrist + $\Delta z$ + `statehist` | 0.0214 | 0.03131 | 0.00863 | 7.41 / 5.23 / 5.01 | 0.130 (0.119) | 10.88 / 7.89 / 5.83 | 20.64 | State `1.67x`, Wrist `1.53x`, **Height $\Delta z$ `1.30x` (Z `1.59x`)**, Hist `1.13x` |
| **Exp 2 (`0026`)** | Rel ($T_o=2$) | Wrist + $z_{\text{foil}}$ + `statehist` | 0.0258 | 0.03063 | 0.00997 | 8.07 / 5.59 / 5.55 | 0.141 (0.134) | 11.16 / 7.62 / 5.39 | 18.89 | Wrist `2.01x`, State `1.48x`, **Height $z_{\text{foil}}$ `1.00x` (ignoriert!)** |
| **Exp 2 (`0027`)** | Abs ($T_o=2$) | Wrist + $\Delta z$ | 0.0160 | 0.02263 | 0.00322 | 8.49 / 6.49 / 5.19 | 0.344 (0.315) | 13.30 / 9.23 / 6.23 | 22.65 | State `4.05x` (XY `34.7x`), Wrist `1.42x`, **Height $\Delta z$ `1.16x` (Z `1.47x`)** |
| **Exp 2 (`0028`)** | Abs ($T_o=2$) | Wrist + $z_{\text{foil}}$ | **0.0155** | 0.02310 | 0.00318 | 9.98 / 7.60 / 6.15 | 0.403 (0.368) | 12.98 / 9.56 / 5.61 | 22.76 | State `3.93x`, Wrist `1.48x`, **Height $z_{\text{foil}}$ `0.97x` (ignoriert!)** |
| **Exp 3 (`0029`)** | Rel ($T_o=4$) | Wrist + $\Delta z$ + `statehist` | 0.0216 | 0.02851 | **0.00699** | 6.50 / 4.67 / 4.30 | 0.113 (0.103) | **8.13 / 7.11 / 3.35** | 19.50 | State `1.80x`, Wrist `1.64x`, **Hist `1.29x`**, **Height $\Delta z$ `1.29x`** |
| **Exp 3 (`0030`)** | Abs ($T_o=4$) | Wrist + $\Delta z$ (nur RGB $T_o=4$) | 0.0161 | **0.02256** | **0.00274** | 9.56 / 7.29 / 5.87 | 0.387 (0.349) | 13.33 / 8.09 / 6.61 | 22.04 | State `4.18x`, Wrist `1.43x`, Height $\Delta z$ `1.14x` (Z `1.41x`) |

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

