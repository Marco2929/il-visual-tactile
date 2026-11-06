# Real-Robot Rollout Matrix (`ROLLOUT.md`)

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

## 2. Noch auf `box_recorded_action.zarr` nachzutrainieren (Exp 1–3 Sieger)

*(Configs unter `third_party/ManiFlow_Policy/ManiFlow/maniflow/config/task/ablation/` sind bereits auf `foil_lifting_box_recorded_action.zarr` umgestellt.)*

### Exp 1: Relative Modalitäts-Ablation mit State-History (`T_o = 2`, `50 ms`)

#### `0022_rel_wrist_pc_statehist` — Relativ: Point Cloud + Wrist-RGB
- **Aktionsraum:** Relativ zum aktuellen Greifer-Frame (17D: $\Delta xyz$ + `rot6d` + 8 Hand-Joints)
- **Sensoren:** Top-Kamera Point Cloud (1024 Punkte `xyz+rgb`) + Wrist-Kamera RGB + volle State-History (`state_history=true`, kein Dropout)
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick für Bild, Punktwolke und Arm-/Hand-State)
- **Zweck im Rollout:** Relative Haupt-Baseline mit Point Cloud (direkter Gegenpol zu `0015`).

#### `0023_rel_wrist_only_statehist` — Relativ: Nur Wrist-RGB (ohne Point Cloud)
- **Aktionsraum:** Relativ zum aktuellen Greifer-Frame (17D)
- **Sensoren:** Nur Wrist-Kamera RGB + volle State-History (`state_history=true`), **keine** Point Cloud, **kein** Höhenskalar
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick)
- **Zweck im Rollout:** Zeigt das Verhalten einer rein lokalen Wrist-Only-Policy ohne jegliche globale Höhenreferenz (Basis für `0025`).

#### `0024_rel_wrist_front_statehist` — Relativ: Wrist-RGB + Front-RGB (ohne Point Cloud)
- **Aktionsraum:** Relativ zum aktuellen Greifer-Frame (17D)
- **Sensoren:** Wrist-Kamera RGB + Front-Kamera RGB + volle State-History (`state_history=true`), **keine** Point Cloud
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick)
- **Zweck im Rollout:** Vollständigkeit der relativen Modalitäts-Ablation (Prüfung, ob `front_rgb` bei relativen Aktionen als Höhen-/Phasen-Anker hilft).

---

### Exp 2: Lokaler 1D-Höhenabstand $\Delta z$ statt Point Cloud (`T_o = 2`, `50 ms`)

#### `0025_rel_wrist_height_delta` — Relativ: Wrist-RGB + 1D-Abstand $\Delta z$
- **Aktionsraum:** Relativ zum aktuellen Greifer-Frame (17D Action, 18D State)
- **Sensoren:** Wrist-Kamera RGB + volle State-History (`state_history=true`) + **1D relativer Höhenabstand zur Folie $\Delta z = z_{\text{ee}} - z_{\text{foil}}$**, **keine** Point Cloud
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick)
- **Zweck im Rollout:** Prüft direkt gegen `0022` (mit PC) und `0023` (ohne PC), ob ein einziger lokaler Distanzwert $\Delta z$ die komplette Top-Kamera-Punktwolke ersetzen kann.

#### `0027_abs_wrist_height_delta` — Absolut: Wrist-RGB + 1D-Abstand $\Delta z$ *(optional)*
- **Aktionsraum:** Absolut im Box-Frame (15D Action, 16D State)
- **Sensoren:** Wrist-Kamera RGB + Propriozeption (`proprio_dropout=0.8`) + **1D relativer Höhenabstand $\Delta z = z_{\text{ee}} - z_{\text{foil}}$**, **keine** Point Cloud
- **Zeitfenster:** $T_o = 2$ Frames ($50\,\text{ms}$ Rückblick)
- **Zweck im Rollout:** Absolutes Gegenstück zu `0025` (Vergleich gegen `0016` Wrist-only und `0015` PC+Wrist).

---

### Exp 3: Zeitfenster-Skalierung auf 4 Schritte (`T_o = 4`, `150 ms`)

#### `0029_rel_wrist_height_delta_to4` — Relativ: Wrist-RGB + 1D-Abstand $\Delta z$ + $150\,\text{ms}$ Historie
- **Aktionsraum:** Relativ zum aktuellen Greifer-Frame (17D Action, 18D State)
- **Sensoren:** Wrist-Kamera RGB + volle State-History (`state_history=true`) + **1D relativer Höhenabstand $\Delta z = z_{\text{ee}} - z_{\text{foil}}$**, **keine** Point Cloud
- **Zeitfenster:** **$T_o = 4$ Frames ($150\,\text{ms}$ Rückblick bei 20 Hz)** für Wrist-RGB ($4 \times 192 = 768$ Tokens) und State ($4 \times 32 = 128\,\text{dim}$)
- **Zweck im Rollout:** Gesamtsieger der Offline-Analyse. Prüft im direkten Vergleich zu `0025` ($T_o=2$, $50\,\text{ms}$), ob das längere $150\,\text{ms}$-Beobachtungsfenster das Geschwindigkeits-Signal stabilisiert und Chunk-Sprünge an den Phasenübergängen (`approach -> search -> transition`) am echten Roboter beseitigt.

