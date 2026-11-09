# Real-Robot Rollout Matrix (`docs/rollouts.md`)

Alle Rollout-Modelle nutzen denselben Ziel-Datensatz **`foil_lifting_box_recorded_action.zarr`** (Teleoperations-Kommando $a_t^{\text{cmd}}$ im Box-Frame bei **20 Hz**, $\Delta t = 50\,\text{ms}$ pro Schritt).

> **Hinweis:** Die alten relativen Läufe ohne State-History (`0019`, `0020`, `0021`) werden **nicht** ausgerollt, da bei ihnen der vergangene Zeitschritt $t-1$ im DiTX-Encoder abgeschnitten wurde (`state_history=false`). Für die relativen Vergleiche werden ausschließlich die korrigierten Läufe mit `state_history=true` verwendet.

---

## 1. Bereits fertig trainiert & analysiert (Absolut-Ablation & Target-Vergleich)

### `0014_abs_wrist_pc` (Target: `state_{t+1}`) — Absolut: Point Cloud + Wrist-RGB auf Folgepose
- **Aktionsraum:** Absolut im Box-Frame (15D: `xyz` + Quaternion `xyzw` + 8 Hand-Joints)
- **Action-Target:** Erreichte Folgepose $a_t = s_{t+1}$ (`foil_lifting_box.zarr`)
- **Sensoren:** Top-Kamera Point Cloud (1024 Punkte `xyz+rgb`) + Wrist-Kamera RGB + Propriozeption (`proprio_dropout=0.8`)
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick)
- **Zweck im Rollout:** Direkter Hardware-Vergleich gegen `0015` zur Prüfung von `state_{t+1}` vs. `recorded_action` ($a_t^{\text{cmd}}$) bei identischer Architektur.
- **Beste Epoche:** **Epoche 30** (`val_action_mse = 0.02349`)
- **Checkpoint:**
  `data/outputs/foil_lifting_wrist-maniflow_multimodal_policy-0014_seed0/checkpoints/epoch=0030-val_action_mse=0.023485.ckpt`

### `0015_abs_wrist_pc` (Target: `recorded_action`) — Absolut: Point Cloud + Wrist-RGB auf Soll-Kommando
- **Aktionsraum:** Absolut im Box-Frame (15D: `xyz` + Quaternion `xyzw` + 8 Hand-Joints)
- **Action-Target:** Teleoperations-Führungsbefehl $a_t^{\text{cmd}}$ (`foil_lifting_box_recorded_action.zarr`)
- **Sensoren:** Top-Kamera Point Cloud (1024 Punkte `xyz+rgb`) + Wrist-Kamera RGB + Propriozeption (`proprio_dropout=0.8`)
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick)
- **Zweck im Rollout:** Klassische Absolut-Baseline mit Point Cloud ohne Schleppfehler.
- **Beste Epoche:** **Epoche 70** (`val_action_mse = 0.03834`)
- **Checkpoint:**
  `data/outputs/foil_lifting_wrist_recorded_action-maniflow_multimodal_policy-0015_seed0/checkpoints/epoch=0070-val_action_mse=0.038337.ckpt`

### `0016_abs_wrist_only` — Absolut: Nur Wrist-RGB (ohne Point Cloud)
- **Aktionsraum:** Absolut im Box-Frame (15D)
- **Sensoren:** Nur Wrist-Kamera RGB + Propriozeption (`proprio_dropout=0.8`), **keine** Point Cloud, **keine** Front-Kamera
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick)
- **Zweck im Rollout:** Prüft im Vergleich zu `0015`, wie stark eine absolute Policy ohne Top-Kamera-Punktwolke in der Höhe (Z) driftet.
- **Beste Epoche:** **Epoche 60** (`val_action_mse = 0.02675`)
- **Checkpoint:**
  `data/outputs/foil_lifting_wrist_recorded_action-maniflow_multimodal_policy-0016_abl_wrist_only_seed0/checkpoints/epoch=0060-val_action_mse=0.026752.ckpt`

### `0017_abs_wrist_front` — Absolut: Wrist-RGB + Front-RGB (ohne Point Cloud)
- **Aktionsraum:** Absolut im Box-Frame (15D)
- **Sensoren:** Wrist-Kamera RGB + Front-Kamera RGB + Propriozeption (`proprio_dropout=0.8`), **keine** Point Cloud
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick)
- **Zweck im Rollout:** Prüft im Vergleich zu `0016`, ob die zweite 2D-Ansicht (`front_rgb`) einen realen Mehrwert gegenüber reinem Wrist-RGB bringt.
- **Beste Epoche:** **Epoche 70** (`val_action_mse = 0.03892`)
- **Checkpoint:**
  `data/outputs/foil_lifting_wrist_recorded_action-maniflow_multimodal_policy-0017_abl_wrist_front_seed0/checkpoints/epoch=0070-val_action_mse=0.038920.ckpt`

---

## 2. Fertig trainiert & evaluiert auf `box_recorded_action.zarr` (Exp 1–3)

Alle Modelle wurden auf demselben Datensatz `foil_lifting_box_recorded_action.zarr` trainiert und synchronisiert.

### Exp 1: Relative Modalitäts-Ablation mit State-History (`T_o = 2`, `50 ms`)

#### `0022_rel_wrist_pc_statehist` — Relativ: Point Cloud + Wrist-RGB
- **Aktionsraum:** Relativ zum aktuellen Greifer-Frame (17D: $\Delta xyz$ + `rot6d` + 8 Hand-Joints)
- **Sensoren:** Top-Kamera Point Cloud (1024 Punkte `xyz+rgb`) + Wrist-Kamera RGB + volle State-History (`state_history=true`, kein Dropout)
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick für Bild, Punktwolke und Arm-/Hand-State)
- **Zweck im Rollout:** Relative Haupt-Baseline mit Point Cloud (direkter Gegenpol zu `0015`).
- **Beste Epoche:** **Epoche 50** (`val_action_mse = 0.03651`, Epoche 70: `0.03694`)
- **Offline-Evaluation:** Full MSE: `0.02852`, XY-Search MSE: `0.00801`, Seam Error Median: `10.28 mm` (P95: `16.85 mm`), 3D Dispersion: `5.82 mm`
- **Checkpoint:**
  `data/outputs/ablation/abl_rel_wrist_pc-ablation/abl_rel_wrist_pc-0022_rel_wrist_pc_statehist_seed0/checkpoints/epoch=0050-val_action_mse=0.036513.ckpt`

#### `0023_rel_wrist_only_statehist` — Relativ: Nur Wrist-RGB (ohne Point Cloud)
- **Aktionsraum:** Relativ zum aktuellen Greifer-Frame (17D)
- **Sensoren:** Nur Wrist-Kamera RGB + volle State-History (`state_history=true`), **keine** Point Cloud, **kein** Höhenskalar
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick)
- **Zweck im Rollout:** Zeigt das Verhalten einer rein lokalen Wrist-Only-Policy ohne jegliche globale Höhenreferenz (Basis für `0025`).
- **Beste Epoche:** **Epoche 70** (`val_action_mse = 0.03223`, Epoche 60: `0.03236`, Epoche 40: `0.03372`)
- **Offline-Evaluation:** Full MSE: `0.03173`, XY-Search MSE: `0.00809`, Seam Error Median: `11.01 mm` (P95: `20.87 mm`), 3D Dispersion: `8.26 mm`
- **Checkpoint:**
  `data/outputs/ablation/abl_rel_wrist-ablation/abl_rel_wrist-0023_rel_wrist_only_statehist_seed0/checkpoints/epoch=0070-val_action_mse=0.032226.ckpt`

#### `0024_rel_wrist_front_statehist` — Relativ: Wrist-RGB + Front-RGB (ohne Point Cloud)
- **Aktionsraum:** Relativ zum aktuellen Greifer-Frame (17D)
- **Sensoren:** Wrist-Kamera RGB + Front-Kamera RGB + volle State-History (`state_history=true`), **keine** Point Cloud
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick)
- **Zweck im Rollout:** Vollständigkeit der relativen Modalitäts-Ablation (Prüfung, ob `front_rgb` bei relativen Aktionen als Höhen-/Phasen-Anker hilft).
- **Beste Epoche:** **Epoche 70** (`val_action_mse = 0.05728`, Epoche 10: `0.04944`)
- **Offline-Evaluation:** Full MSE: `0.02484`, XY-Search MSE: `0.00930`, Seam Error Median: `10.57 mm` (P95: `19.36 mm`), 3D Dispersion: `6.95 mm`
- **Checkpoint:**
  `data/outputs/ablation/abl_rel_wrist_front-ablation/abl_rel_wrist_front-0024_rel_wrist_front_statehist_seed0/checkpoints/epoch=0070-val_action_mse=0.057283.ckpt`

---

### Exp 2: Lokaler 1D-Höhenabstand $\Delta z$ statt Point Cloud (`T_o = 2`, `50 ms`)

#### `0025_rel_wrist_height_delta` — Relativ: Wrist-RGB + 1D-Abstand $\Delta z$
- **Aktionsraum:** Relativ zum aktuellen Greifer-Frame (17D Action, 18D State)
- **Sensoren:** Wrist-Kamera RGB + volle State-History (`state_history=true`) + **1D relativer Höhenabstand zur Folie $\Delta z = z_{\text{ee}} - z_{\text{foil}}$**, **keine** Point Cloud
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick)
- **Zweck im Rollout:** Prüft direkt gegen `0022` (mit PC) und `0023` (ohne PC), ob ein einziger lokaler Distanzwert $\Delta z$ die komplette Top-Kamera-Punktwolke ersetzen kann.
- **Beste Epoche:** **Epoche 60** (`val_action_mse = 0.03075`, Epoche 70: `0.03083`, Epoche 40: `0.03295`)
- **Offline-Evaluation:** Full MSE: `0.03131`, XY-Search MSE: `0.00863`, Seam Error Median: `10.88 mm` (P95: `20.64 mm`), 3D Dispersion: `7.41 mm`
- **Checkpoint:**
  `data/outputs/ablation/abl_rel_wrist_height_delta-ablation/abl_rel_wrist_height_delta-0025_rel_wrist_height_delta_seed0/checkpoints/epoch=0060-val_action_mse=0.030753.ckpt`

#### `0027_abs_wrist_height_delta` — Absolut: Wrist-RGB + 1D-Abstand $\Delta z$
- **Aktionsraum:** Absolut im Box-Frame (15D Action, 16D State)
- **Sensoren:** Wrist-Kamera RGB + Propriozeption (`proprio_dropout=0.8`) + **1D relativer Höhenabstand $\Delta z = z_{\text{ee}} - z_{\text{foil}}$**, **keine** Point Cloud
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick)
- **Zweck im Rollout:** Absolutes Gegenstück zu `0025` (Vergleich gegen `0016` Wrist-only und `0015` PC+Wrist).
- **Beste Epoche:** **Epoche 60** (`val_action_mse = 0.01597`, Epoche 50: `0.01600`)
- **Offline-Evaluation:** Full MSE: `0.02263`, XY-Search MSE: `0.00322`, Seam Error Median: `13.30 mm` (P95: `22.65 mm`), 3D Dispersion: `8.49 mm`
- **Checkpoint:**
  `data/outputs/ablation/abl_abs_wrist_height_delta-ablation/abl_abs_wrist_height_delta-0027_abs_wrist_height_delta_seed0/checkpoints/epoch=0060-val_action_mse=0.015972.ckpt`

---

### Exp 3: Zeitfenster-Skalierung auf 4 Schritte (`T_o = 4`, `150 ms`)

#### `0029_rel_wrist_height_delta_to4` — Relativ: Wrist-RGB + 1D-Abstand $\Delta z$ + $150\,\text{ms}$ Historie *(Gesamtsieger Relativ)*
- **Aktionsraum:** Relativ zum aktuellen Greifer-Frame (17D Action, 18D State)
- **Sensoren:** Wrist-Kamera RGB + volle State-History (`state_history=true`) + **1D relativer Höhenabstand $\Delta z = z_{\text{ee}} - z_{\text{foil}}$**, **keine** Point Cloud
- **Zeitfenster:** **$T_o = 4$ Frames ($150\,\text{ms}$ Rückblick bei 20 Hz)** für Wrist-RGB ($4 \times 192 = 768$ Tokens) und State ($4 \times 32 = 128\,\text{dim}$)
- **Zweck im Rollout:** **Gesamtsieger der Offline-Analyse.** Reduziert den Nahtfehler auf das absolute Minimum (`8.13 mm`, -26% vs. $T_o=2$). Prüft, ob das längere $150\,\text{ms}$-Beobachtungsfenster das Geschwindigkeits-Signal stabilisiert und Chunk-Sprünge an den Phasenübergängen (`approach -> search -> transition`) am echten Roboter beseitigt.
- **Beste Epoche:** **Epoche 60** (`val_action_mse = 0.03065`, Epoche 70: `0.03066`)
- **Offline-Evaluation:** Full MSE: `0.02851`, XY-Search MSE: `0.00699`, Seam Error Median: **`8.13 mm`** (P95: `19.50 mm`), 3D Dispersion: `6.50 mm`
- **Checkpoint:**
  `data/outputs/ablation/abl_rel_wrist_height_delta_to4-ablation/abl_rel_wrist_height_delta_to4-0029_rel_wrist_height_delta_to4_seed0/checkpoints/epoch=0060-val_action_mse=0.030649.ckpt`

---

## 3. Progress Head Checkpoints (Erfolgs- & Abbruch-Prädiktion)

Alle Progress-Heads wurden mit dem stratifizierten 70/30-Split über 30 Epochen trainiert. Der Progress-Kopf schätzt synchron zur Policy die relative Trajektorienfortschritts-Metrik $\hat{p}_t \in [0, 1]$.

| Run ID | Name | Glättung ($\lambda_{\text{smooth}}$) | $\Delta_{\max}$ | Beste Epoche | Val Loss (Huber) | Val MAE | FPR ($p > 0.9$ bei $< 70\%$ traj) | Checkpoint-Pfad |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`prg_0001`** | `baseline_v2_clean` | **0.00** | 0.03 | **Epoche 3** | **0.07832** | **0.09598** | **0.78%** | `data/outputs/progress_heads/prg_0001_baseline_v2_clean/best_progress_head.pt` |
| **`prg_0002`** | `smooth_w005` | 0.05 | 0.03 | Epoche 14 | 0.08300 | 0.10037 | 1.82% | `data/outputs/progress_heads/prg_0002_smooth_w005/best_progress_head.pt` |
| **`prg_0003`** | `smooth_w010` | 0.10 | 0.02 | Epoche 3 | 0.14359 | 0.16403 | 0.82% | `data/outputs/progress_heads/prg_0003_smooth_w010/best_progress_head.pt` |

> **Empfehlung für Rollout:** **`prg_0001_baseline_v2_clean`** liefert mit $0.0783$ Huber-Loss, $0.096$ MAE und nur $0.78\%$ Fehlalarmquote den präzisesten und reaktionsschnellsten Abbruch-Trigger für den Live-Rollout.

---

## 4. Empfohlene Hardware-Rollout-Reihenfolge

1. **Haupt-Lineup (Produktiv-Kandidat):**
   - **Policy:** `0029_rel_wrist_height_delta_to4` (Epoche 60, `8.13 mm` Nahtfehler)
   - **Progress Head:** `prg_0001_baseline_v2_clean` (Epoche 3)
   - **Begründung:** Höchste Glätte an Nahtstellen, minimale Drift und sauberer Abbruch ohne False-Positive-Spikes.

2. **Ablation 1 (Einfluss Zeithistorie $T_o=2$ vs. $T_o=4$):**
   - **Policy:** `0025_rel_wrist_height_delta` (Epoche 70, $T_o=2$, $50\,\text{ms}$)
   - **Begründung:** Direkter Gegentest zu `0029` mit identischen Modalitäten (Wrist + $\Delta z$), um den realen Gewinn der $150\,\text{ms}$-Historie am Manipulator zu isolieren.

3. **Ablation 2 (Einfluss Point Cloud vs. 1D-Höhenabstand $\Delta z$):**
   - **Policy:** `0022_rel_wrist_pc_statehist` (Epoche 50, PC + Wrist)
   - **Begründung:** Zeigt, ob die schwere 3D-Punktwolke auf echter Hardware einen Vorteil gegenüber dem leichten 1D-Höhenskalar bietet oder eher Rechenlast/Latenz erhöht.

4. **Absolut-Baseline:**
   - **Policy:** `0015_abs_wrist_pc` (Epoche 70)
   - **Begründung:** Klassische Absolut-Policy mit globaler Koordinate als Referenzpunkt gegen die relativen Modelle.

