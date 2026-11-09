# Auswertung: Höhen-Konditionierung, Propriozeptions-Historie & Progress Heads

> **Datensatz-Grundlage:** Alle hier dokumentierten Modelle wurden auf dem aktuellen Hauptdatensatz **`foil_lifting_box_recorded_action.zarr`** (306 Episoden, 54.752 Zeitschritte, echte vorauslaufende Teleop-Kommandos) bzw. **`foil_lifting_box_progress.zarr`** (bereinigte `v2_clean`-PCHIP-Fortschrittsfunktion) trainiert und evaluiert.
> Der alte Datensatz `foil_lifting_wrist_tesollo.zarr` (80 Demos, `next_state`) wird für neue Modelle nicht mehr verwendet.

---

## 1. Auswertung der neuen Policy-Architekturen (`0022`–`0030`)

In dieser Versuchsreihe wurden drei gezielte Modifikationen gegenüber den Baselines (`0015`–`0021`) untersucht:
1. **Propriozeptions-Historie (`state_history = true` in `0022`, `0023`, `0024`):**
   Alle $T_o$ Zeitschritte werden im State-Token übergeben. Bei relativen Modellen ist der Schritt $t < T_o - 1$ relativ zum aktuellen Anker $t = T_o - 1$ kodiert, liefert also direkt den Geschwindigkeits- und Anflugvektor.
2. **Skalare Höhenkonditionierung (`height_delta` in `0025`, `0027` vs. `height_abs` in `0026`, `0028`):**
   Ersatz der 3D-Punktwolke durch einen aus der Punktwolke extrahierten skalaren Höhenwert ($\Delta z = z_{\text{EE}} - z_{\text{foil}}$ vs. absolute Folienhöhe $z_{\text{foil}}$).
3. **Beobachtungshorizont $T_o = 4$ (`0029`, `0030`):**
   Verdopplung des Zeithorizonts von $T_o = 2$ ($67\,\text{ms}$) auf $T_o = 4$ ($133\,\text{ms}$) in Kombination mit `wrist_rgb + height_delta + state_history`.

---

### 1.1 Validierungs-MSE und Modalitäts-Sensitivität auf `box_recorded_action.zarr`

Die Evaluation erfolgt auf einem stratifizierten Validierungs-Set ($N=192$ Fenster, gleichmäßig aufgeteilt über alle 4 Aufgabenphasen `approach`, `search`, `transition`, `hold`).
Die Sensitivitätsfaktoren geben den Faktor an, um den der Vorhersagefehler bei Shuffling des jeweiligen Eingangssignals ansteigt ($1.0\times$ = ignoriert, $>1.5\times$ = stark genutzt).

| Modell-ID | Architektur & Features | Best Ckpt MSE | Strat. `full` MSE | `xy_search` MSE | `shuffle_wrist` (`xy_s`) | `shuffle_pc` / `front` (`xy_s`) | `shuffle_state` (`xy_s`) | `state_hist` vs. `state_curr` | `height` vs. `pose_no_h` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0015_abs_pc_wrist`** | Abs, PC + Wrist (`hist=F`) | 0.03834 | 0.04462 | 0.01493 | 1.60x (`1.10x`) | PC: 1.56x (`3.53x`) | 2.13x (`6.21x`) | — | — |
| **`0016_abs_wrist_only`** | Abs, Wrist-Only (`hist=F`) | 0.02675 | 0.03197 | 0.01105 | 2.47x (`1.67x`) | — | 3.55x (`12.28x`) | — | — |
| **`0017_abs_wrist_front`** | Abs, Wrist + Front (`hist=F`) | 0.03892 | 0.04579 | 0.01338 | 1.37x (`1.31x`) | Fr: 1.32x (`1.89x`) | 2.16x (`8.68x`) | — | — |
| **`0027_abs_wrist_h_delta`** | **Abs, Wrist + $\Delta z$ (`hist=F`)** | **0.02836** | **0.03220** | **0.00838** | 1.31x (`1.84x`) | — | 2.29x (`13.73x`) | — | **H: 1.44x** / P: 1.88x |
| **`0019_rel_pc_wrist`** | Rel, PC + Wrist (`hist=F`) | 0.03745 | 0.04145 | 0.02350 | 1.72x (`1.92x`) | PC: 1.28x (`1.18x`) | 1.15x (`0.97x`) | — | — |
| **`0022_rel_pc_wrist_sh`** | **Rel, PC + Wrist (`hist=T`)** | **0.03651** | **0.03923** | **0.01986** | 1.45x (`1.19x`) | PC: 1.31x (`1.16x`) | 1.15x (`0.98x`) | Hist: 1.02x / Curr: 1.10x | — |
| **`0020_rel_wrist_only`** | Rel, Wrist-Only (`hist=F`) | 0.03129 | 0.03425 | 0.01696 | **2.81x** (`2.45x`) | — | 1.19x (`0.89x`) | — | — |
| **`0023_rel_wrist_only_sh`** | **Rel, Wrist-Only (`hist=T`)** | 0.03223 | **0.03369** | 0.01741 | 1.96x (`1.47x`) | — | 1.25x (`0.95x`) | Hist: 1.01x / Curr: 1.15x | — |
| **`0021_rel_wrist_front`** | Rel, Wrist + Front (`hist=F`) | 0.05464 | 0.04579 | 0.02096 | 1.56x (`1.86x`) | Fr: 1.17x (`1.03x`) | 1.15x (`0.92x`) | — | — |
| **`0024_rel_wrist_front_sh`**| **Rel, Wrist + Front (`hist=T`)** | 0.05728 | **0.04485** | **0.01715** | 1.32x (`1.17x`) | Fr: 1.04x (`1.42x`) | 1.11x (`1.09x`) | Hist: 1.02x / Curr: 1.04x | — |
| **`0025_rel_wrist_h_delta`** | **Rel, Wrist + $\Delta z$ (`hist=T, To=2`)**| 0.03075 | **0.02973** | **0.01290** | 1.63x (`1.99x`) | — | **1.83x** (`1.36x`) | **Hist: 1.26x** / Curr: 1.34x | **H: 1.46x** / P: 1.34x |
| **`0029_rel_wrist_h_d_to4`** | **Rel, Wrist + $\Delta z$ (`hist=T, To=4`)**| **0.03065** | **0.02963** | 0.01570 | 1.59x (`1.68x`) | — | **1.84x** (`1.22x`) | **Hist: 1.30x** / Curr: 1.15x | **H: 1.40x** / P: 1.35x |

---

### 1.2 Phasen- und Achsen-Zerlegung der Sub-Channels

| Modell | Ablations-Kanal | `approach` | `search` | `transition` | `hold` | Achse `xy` | Achse `z` | Achse `rot` | Achse `hand` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0022_rel_pc_wrist_sh`** | `shuffle_wrist` | **2.01x** | 1.11x | 0.99x | **2.03x** | **1.58x** | 1.29x | 1.48x | 1.43x |
| | `shuffle_pc` | 1.43x | 1.16x | 1.10x | **1.95x** | 1.33x | **1.47x** | 1.28x | 1.32x |
| | `shuffle_state_hist` / `curr` | 0.97x / 0.98x | 0.99x / **1.20x** | 1.11x / 1.13x | 1.00x / 1.05x | 1.01x / 0.99x | 0.96x / 0.95x | 0.99x / 0.96x | 1.04x / **1.18x** |
| **`0023_rel_wrist_only_sh`** | `shuffle_wrist` | **2.74x** | **1.43x** | **1.54x** | **1.89x** | **2.13x** | 1.40x | **2.00x** | **1.96x** |
| | `shuffle_state_hist` / `curr` | 0.95x / 1.02x | 0.98x / **1.17x** | 1.14x / **1.27x** | 1.01x / 1.18x | 1.00x / 1.01x | 0.94x / 0.89x | 1.01x / 1.02x | 1.02x / **1.25x** |
| **`0025_rel_wrist_h_delta`** | `shuffle_wrist` | **1.55x** | **1.57x** | **1.84x** | **1.64x** | **1.97x** | **1.68x** | **1.75x** | **1.54x** |
| ($T_o=2$) | `shuffle_state_hist` | **1.42x** | 1.09x | **1.33x** | 1.05x | **1.28x** | 1.08x | 1.15x | **1.32x** |
| | `shuffle_state_curr` | **1.43x** | **1.29x** | **1.34x** | 1.25x | **1.27x** | 1.03x | 1.13x | **1.48x** |
| | **`shuffle_height` ($\Delta z$)** | **2.10x** | 1.05x | 1.07x | 1.22x | **1.52x** | **1.27x** | 1.24x | **1.58x** |
| | `shuffle_pose_no_height` | 1.12x | **1.37x** | **1.52x** | **1.51x** | 1.24x | 1.01x | 1.09x | **1.50x** |
| **`0029_rel_wrist_h_d_to4`** | `shuffle_wrist` | **1.60x** | **1.42x** | **1.63x** | **1.74x** | **1.76x** | **1.46x** | **1.75x** | **1.50x** |
| ($T_o=4$) | **`shuffle_state_hist`** | **1.64x** | 1.21x | 1.22x | 0.92x | **1.34x** | **1.14x** | 1.20x | **1.37x** |
| | `shuffle_state_curr` | 1.21x | 1.23x | 1.27x | 0.75x | 1.03x | 0.97x | 1.06x | 1.23x |
| | **`shuffle_height` ($\Delta z$)** | **2.06x** | 1.06x | 1.01x | 1.20x | **1.43x** | **1.23x** | 1.21x | **1.52x** |
| | `shuffle_pose_no_height` | 1.07x | **1.48x** | **1.58x** | **1.37x** | 1.13x | 0.99x | 1.12x | **1.53x** |
| **`0027_abs_wrist_h_delta`** | `shuffle_wrist` | **1.36x** | 1.15x | 1.15x | **1.85x** | **1.49x** | **1.40x** | 1.26x | 1.30x |
| | **`shuffle_height` ($\Delta z$)** | **2.36x** | 0.99x | 1.05x | 1.10x | 1.03x | **1.43x** | 1.26x | **1.53x** |
| | `shuffle_pose_no_height` | 1.90x | **1.76x** | **1.73x** | **2.44x** | **8.75x** | **1.82x** | 1.58x | 1.45x |

---

### 1.3 Vorhersage-Konsistenz ($K=8$ Rausch-Seeds bei identischer Beobachtung) & Chunk-Seam ($\Delta t = 8$)

| Modell | Norm. `std_xy` | Rel. Spread `std_xy / gt` | `search` `std_xy` | Phys. 3D STD (`mm`) | Phys. XY STD (`mm`) | Phys. Z STD (`mm`) | Max Pair Dist (`mm`) | Seam 3D Med (`mm`) | Seam XY / Z Med (`mm`) | Seam Phasen (`app` / `srch` / `tran` / `hold`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0015_abs_pc_wrist`** | 0.0119 | 0.224 | 0.0108 | **18.43** | **9.73** | 14.92 | **51.57** | **30.82** | 18.88 / 17.51 | 28.4 / 23.0 / 34.4 / 51.7 |
| **`0016_abs_wrist_only`** | 0.0153 | 0.286 | 0.0155 | 25.00 | 13.68 | 19.67 | 69.51 | 41.61 | 26.86 / 23.95 | 38.0 / 30.8 / 45.7 / 62.3 |
| **`0017_abs_wrist_front`** | 0.0127 | 0.238 | 0.0117 | 22.84 | 10.98 | 19.22 | 64.10 | 34.33 | 20.66 / 20.06 | 30.2 / 24.4 / 37.3 / 62.0 |
| **`0027_abs_wrist_h_delta`** | **0.0144** | **0.259** | **0.0132** | **21.04** | **11.02** | **16.66** | **58.23** | **39.52** | **25.79 / 22.24** | 34.9 / 34.6 / 51.8 / **39.5** |
| **`0019_rel_pc_wrist`** | **0.0165** | **0.142** | **0.0146** | **20.47** | **12.19** | **15.58** | **55.90** | **41.66** | 25.49 / 26.64 | 31.3 / 36.1 / 46.8 / 67.1 |
| **`0022_rel_pc_wrist_sh`** | 0.0233 | 0.200 | 0.0214 | 28.35 | 16.74 | 21.61 | 77.86 | 46.32 | 29.58 / 30.70 | 40.2 / 40.1 / 69.7 / 53.0 |
| **`0020_rel_wrist_only`** | 0.0249 | 0.214 | 0.0236 | 30.69 | 18.88 | 22.89 | 84.51 | 56.48 | 36.02 / 31.35 | 51.6 / 47.0 / 61.1 / 73.1 |
| **`0023_rel_wrist_only_sh`** | **0.0212** | **0.182** | **0.0203** | **25.74** | **15.75** | **19.30** | **70.87** | **48.93** | **32.56 / 29.44** | 46.2 / **35.8** / 62.0 / 59.1 |
| **`0021_rel_wrist_front`** | 0.0365 | 0.314 | 0.0365 | 41.31 | 26.37 | 30.39 | 113.10 | 61.85 | 41.64 / 37.89 | 61.4 / 53.7 / 59.8 / 79.8 |
| **`0024_rel_wrist_front_sh`**| **0.0181** | **0.155** | **0.0168** | **22.03** | **13.42** | **16.59** | **60.81** | **49.86** | **26.43 / 31.34** | 36.9 / 36.5 / 68.2 / 67.7 |
| **`0025_rel_wrist_h_delta`** | 0.0225 | 0.193 | 0.0222 | 27.34 | 17.65 | 19.80 | 75.08 | 51.97 | 33.09 / 30.71 | 45.6 / 38.3 / 58.1 / 61.5 |
| **`0029_rel_wrist_h_d_to4`** | **0.0213** | **0.176** | **0.0200** | **24.90** | **15.53** | **18.35** | **68.15** | **51.08** | **31.26 / 29.00** | 46.5 / **37.4** / 73.7 / 58.9 |

---

### 1.4 Wesentliche Erkenntnisse

1. **`height_delta` ($\Delta z = z_{\text{EE}} - z_{\text{foil}}$) schlägt die Punktwolke:**
   * **Phasenspezifische Nutzung:** In `0025`, `0027` und `0029` nutzt das Netzwerk $\Delta z$ **ausschließlich in der `approach`-Phase** (`shuffle_height = 2.06x–2.36x`), um das Abbremsen vor dem Tischkontakt zu steuern. Nach dem Kontakt sinkt der Wert auf `1.01x–1.07x` ab; die Steuerung wird lückenlos an `wrist_rgb` (`1.42x–1.84x`) und die Fingerpropriozeption übergeben.
   * **Geringerer MSE:** Auf `box_recorded_action.zarr` erzielt `0025_rel_wrist_height_delta` einen um **24 % geringeren Gesamt-MSE** (`0.0297` vs. `0.0392`) und einen um **35 % geringeren `xy_search`-MSE** (`0.0129` vs. `0.0199`) als das Point-Cloud-Modell `0022`.
   * **Absolute Höhe ($z_{\text{foil}}$) ist wirkungslos:** In `0026` und `0028` lag `shuffle_height` bei `0.97x–1.01x`. Weil $z_{\text{foil}}$ pro Episode konstant bleibt und bei relativen Posen $z_{\text{EE}}$ nicht absolut vorliegt, fehlt der Policy die Abstands-Information.

2. **Wirkung der Propriozeptions-Historie (`state_history = true`):**
   * Ohne Höhenskalar (`0022`, `0023`, `0024`) bleibt `state_hist` bei $T_o=2$ ein schwaches Einzelsignal (`1.01x–1.02x`), stabilisiert jedoch die Policy indirekt: Bei `0023` sinkt die räumliche 3D-Streuung von `30.7 mm` auf `25.7 mm`; bei `0024` wird die extreme Front-Kamera-Instabilität von `41.3 mm` auf `22.0 mm` halbiert.
   * In Kombination mit `height_delta` und $T_o=4$ (`0029_rel_wrist_height_delta_to4`) wird `state_hist` im Anflug massiv aktiv (**`1.64x` in `approach`**) und übertrifft dort `state_curr` (`1.21x`), da über $133\,\text{ms}$ der Sink- und Anflugvektor abgeleitet werden kann.

3. **Vorteil des erweiterten Zeithorizonts ($T_o = 4$):**
   * `0029_rel_wrist_height_delta_to4` senkt gegenüber $T_o=2$ (`0025`) die 3D-Sample-Streuung bei identischer Beobachtung von `27.34 mm` auf **`24.90 mm`** (`xy_std` von `17.65 mm` auf **`15.53 mm`**, P95-Max-Pair-Distanz von `108.3 mm` auf **`90.0 mm`**).

---

## 2. Auswertung der Progress Heads (`prg_0001`–`prg_0003`)

Trainiert über 30 Epochen auf `progress_split_70_30.json` (647 Episoden: 453 Train / 194 Val) mit DINOv3-Wrist-Patch-Tokens und `v2_clean`-PCHIP-Labels.

| Run-ID | Glättungs-Hyperparameter | Best Epoch | `best_val_loss` | `best_val_mae` | **`best_fpr_09`** (False Positives $\ge 0.9$ bei GT $< 0.7$) | Bewertung |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`prg_0001_baseline_v2_clean`** | `smooth_weight = 0.0`, `delta_max = 0.03` | 3 | **0.07832** | **0.09598** | **0.00782 (0.78 %)** | **Bester Punktschätzer:** Niedrigster MAE (`9.6 %`) und geringste Fehlalarmrate (`0.78 %`). Ohne zeitliche Strafe neigt das Training nach Epoche 3 zu leichtem Overfitting. |
| **`prg_0002_smooth_w005`** | `smooth_weight = 0.05`, `delta_max = 0.03` | **14** | 0.08300 | 0.10037 | 0.01824 (1.82 %) | **Beste zeitliche Stabilität:** Kaum Verlust beim MAE (`10.0 %` vs. `9.6 %`), trainiert stabil über 14 Epochen und dämpft hochfrequentes Zappeln in der Vorhersage. |
| **`prg_0003_smooth_w010`** | `smooth_weight = 0.10`, `delta_max = 0.02` | 3 | 0.14359 | 0.16403 | 0.00817 (0.82 %) | **Über-regularisiert:** Zu enge Toleranz (`delta_max = 0.02`) bei hoher Gewichtung (`0.10`) bremst reale Anstiege aus; MAE verschlechtert sich um 70 %. |

---

### 2.1 Der genaue Unterschied zwischen `prg_0002` und `prg_0003`

In `train_progress_head.py` wird der temporale Regularisierungsterm wie folgt berechnet:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Huber}}(\hat{y}_t, y_t) + w_{\text{smooth}} \cdot \frac{1}{B} \sum_{i=1}^B \left( \max\left(0, |\hat{y}_{t+1} - \hat{y}_t| - \Delta_{\max}\right) \right)^2$$

* **`prg_0002_smooth_w005` ($w_{\text{smooth}} = 0.05, \Delta_{\max} = 0.03$):**
  * Erlaubt bis zu **3 % Fortschrittszuwachs pro Frame** ($67\,\text{ms}$) völlig straffrei.
  * Erst bei Sprüngen $> 3 \%$ greift eine moderate quadratische Strafe mit Gewicht `0.05`.
  * *Verhalten:* Reale steile Phasenanstiege (z. B. schnelles Anheben der Folie) werden unverfälscht durchgelassen, während unrealistisches Rauschen effektiv unterdrückt wird. Der Schätzfehler bleibt mit `MAE = 0.100` praktisch gleichauf mit der ungedämpften Baseline (`0.096`).
* **`prg_0003_smooth_w010` ($w_{\text{smooth}} = 0.10, \Delta_{\max} = 0.02$):**
  * Toleriert nur noch **2 % Zuwachs pro Frame** (eine um 33 % engere Toleranz).
  * Bei Überschreitung greift eine **doppelt so hohe Verlustgewichtung (`0.10`)**.
  * *Verhalten:* Ein realer Foliengriff und Hub erzeugt in der Praxis typischerweise Zuwächse von $3\dots 5 \%$ pro Zeitschritt. Die Strafe zwingt das Modell, steilen realen Flanken hinterherzuhinken. Dadurch explodiert der `val_mae` auf `0.164` (+70 % Fehler), und das Training bricht nach Epoche 3 ein.

---

## 3. Empfehlungen für Rollouts am Roboter

1. **Beste Relativ-Policy: `0029_rel_wrist_height_delta_to4` (bzw. `0025_rel_wrist_height_delta` bei $T_o=2$):**
   * Höchste Zielorientierung über `wrist_rgb` in der Suchphase (`1.68x–1.99x`).
   * Sauberes Abbremsen in `approach` über $\Delta z$ (`2.06x–2.10x`).
   * Geringste Vorhersage-Streuung im relativen Raum (`24.9 mm` 3D-STD, `15.5 mm` XY-STD).
2. **Beste Absolut-Policy: `0027_abs_wrist_height_delta`:**
   * Schlägt das Point-Cloud-Modell `0015` im MSE um 28 % und im `xy_search`-MSE um 44 % bei praktisch gleicher lateraler Stabilität (`11.0 mm` vs. `9.7 mm` XY-STD).
3. **Progress Head für Rollouts:**
   * **`prg_0001_baseline_v2_clean`** für punktgenaue Schwellenwert-Auslösung (`FPR@0.9 = 0.78 %`).
   * **`prg_0002_smooth_w005`** falls im Live-Betrieb eine monotonere, gefilterte Trajektorie ohne Zitter-Ausschläge benötigt wird.
