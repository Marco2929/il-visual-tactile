# Rollout Höhen-Setup & Evaluierungs-Referenz

Kompakte Übersicht über den physischen Aufbau der drei Teststufen, die gemessene Testhöhe und den direkten Abgleich mit den Datensätzen (B-Säule & A-Säule).

---

## 1. Unterste Schicht (`folien_unten` / Prompt: `3` "Palette untere Hoehe")
* **Physischer Aufbau:** **1 Schicht B-Säulen** liegt in der Kiste, Folie liegt darauf.
* **Gemessene Testhöhe:** **$z \approx -413 \dots -423\,\text{mm}$** (Live: **$-413.1\,\text{mm}$**)
* **Vergleich mit Datensatz (unterste Lage L3):**
  * **B-Säule:** **Exakte Punktlandung!** (Datensatz: $-416.7\,\text{mm}$, Abweichung nur **$+3.6\,\text{mm}$**).
  * **A-Säule:** **Optimal im Trainingsbereich** (Datensatz: $-431.3\,\text{mm}$, Abweichung nur **$+1.8\,\text{cm}$**, voll im Toleranzband von $-460 \dots -411\,\text{mm}$).

---

## 2. Mittlere Schicht (`folien_mitte` / Prompt: `2` "Palette Zwischenhoehe")
* **Physischer Aufbau:** **Noch eine Schicht B-Säulen** draufgelegt (insgesamt 2 Schichten B-Säulen), Folie darauf.
* **Gemessene Testhöhe:** **$z \approx -360\,\text{mm}$** (ca. $+5.5\,\text{cm}$ über unten).
* **Vergleich mit Datensatz (unterste Lage L3):**
  * **B-Säule:** **Exakte Punktlandung!** (Datensatz: $-354.9\,\text{mm}$, Abweichung nur **$-5.1\,\text{mm}$**).
  * **A-Säule:** Liegt ca. **$6\,\text{cm}$ tiefer** als im Training (Datensatz: $-299.0\,\text{mm}$).
  * **Eval-Ziel:** Testet gezielt die **Höhen-Generalisierung** der Policy (Prüfen, ob die Policy die A-Säule auch $6\,\text{cm}$ tiefer zuverlässig greift).

---

## 3. Oberste Schicht (`folien_oben` / Prompt: `1` "Palette Nominalhoehe")
* **Physischer Aufbau:** Eine Schicht weniger als unten (also **alle Teile raus**), **24 cm Monitor-Karton** hineinstellen, Folie oben auflegen.
* **Erwartete Testhöhe:** **$z \approx -237 \dots -242\,\text{mm}$** (Live gemessen mit Monitor-Karton am Boden: **$-242.3\,\text{mm}$**; bzw. mit Inlay/Zwischenlage bis **$-186\,\text{mm}$**).
* **Vergleich mit Datensatz (unterste Lage L3):**
  * **A-Säule:** **Exakter Volltreffer!** (Datensatz: $-186.5\,\text{mm}$, Abweichung nahezu **$0\,\text{cm}$**).
  * **B-Säule:** Liegt im oberen Bereich (Datensatz: $-158.0\,\text{mm}$, Abweichung ca. $-2.8\,\text{cm}$).

---

## Rollout-Prompts im Terminal
* `1` $\to$ **Palette Nominalhoehe** (oberste Schicht)
* `2` $\to$ **Palette Zwischenhoehe** (mittlere Schicht)
* `3` $\to$ **Palette untere Hoehe** (unterste Schicht)

### Spezielle Test-Varianten:
* Links / Mitte / Rechts: `"Palette Nominalhoehe links|mitte|rechts"`
* Retry bei Abrutschen: `"Palette Nominalhoehe retry"`
* Höhenvariation: `"Palette Nominalhoehe increased"` / `"Palette Nominalhoehe lowered"`

---

## Schnellcheck vor Teststart
```bash
python3 data/rollouts/tools/compare_foil_height.py
```
*(Zeigt die ZED-2i Live-Höhe mit direktem L3-Vergleich an).*

---

## 4. Evaluierungs-Matrix & Test-Katalog

### 2-Tier Evaluierungs-Strategie
* **Tier 1 (Grobscreening, $N=1$ Run pro Bedingung, ~15–20 min pro Modell):**
  * Schnelle Validierung aller Checkpoints über den Test-Katalog mit $N=1$.
  * Ziel: Schnelle Ausmusterung von Modellen mit grundlegenden Fehlern (z. B. Drift, kein Greifimpuls, Verfehlen der Box).
* **Tier 2 (Tiefen-Benchmark, $N=3$ Runs pro Bedingung, ~45–60 min pro Modell):**
  * Vertiefte statistische Absicherung nur für die vielversprechendsten Kandidaten (Success Rate $0/3, 1/3, 2/3, 3/3$).
  * Ziel: Belastbare Zahlen für Auswertung/Publikation und Ermittlung von Robustheit gegen Störungen und Reaktivität.

### Test-Katalog & Versuchsmatrix ($N=3$, Anker: Mitte / P1 / A-Säule / L1)

| Bereich | Testfall / Variation | Spezifische Bedingung | Runs ($N=1$) | Runs ($N=3$) | Erfolgskriterium |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **0. Anker** | Referenz-Setup | $P1$ (Center), Höhe Mitte, A-Säule, Lage $L1$ | 1 | 3 | Hebt Folie sauber über Schwellenhöhe |
| **1. Höhen** | 3 Höhen | Unten ($z \approx -413\,\text{mm}$), *(Mitte)*, Oben ($z \approx -240\,\text{mm}$) | +2 | +6 | Z-Höhen-Generalisierung ohne Box-Kollision |
| **2. Positionen** | 5 Startpositionen | *(P1 Center)*, $P2$ (VL), $P3$ (VR), $P4$ (HL), $P5$ (HR) | +4 | +12 | XY-Workspace-Invarianz & Folienfindung |
| **3. Bauteile** | 3 Geometrien | *(A-Säule)*, B-Säule, Bauteil 3 | +2 | +6 | Visuelle & taktile Invarianz ggü. Werkstück |
| **4. Schichten** | 3 Folienlagen | *(L1 oben)*, L2 (Zwischenlage), L3 (Bodenfolie) | +2 | +6 | Vereinzelung & Tiefenstaffelung |
| **5. Reaktivität** | Störungs-/Retry-Test | $P1$, Mitte (Folie wird bei $z_{50\%}$ mutwillig entzogen) | +1 | +3 | Erkennt Folienverlust & setzt zum Neugreifen an |
| **Gesamt** | **Kompletter Benchmark** | **Alle 5 Testachsen** | **12 Runs** | **36 Runs** | **Dauer: ~15 Min (Tier 1) bzw. ~50 Min (Tier 2)** |

### In-Episode Abbruch- & Bewertungskriterien
* **Maximales Zeitfenster (Timeout):** 20 Sekunden (400 Schritte bei 20 Hz).
* **Greifversuche (Retries pro Run):** Maximal 2 Versuche.
  * Erster Griff daneben, setzt selbstständig nach und greift innerhalb von 20s $\rightarrow$ **Erfolg mit Retry**.
  * Nach 2 Fehlversuchen oder 20s keine Folie gegriffen $\rightarrow$ **Fehlschlag (Fail)**.
* **Reaktivitäts-Bewertung:**
  * **Stufe 1 (Voll reaktiv):** Erkennt Folienentzug, bricht Aufwärtsbewegung ab, senkt sich ab und greift erneut.
  * **Stufe 2 (Teil-reaktiv):** Reagiert auf Entzug (z. B. Hand schließt/öffnet), findet Folie jedoch nicht vor Timeout.
  * **Stufe 3 (Open-Loop):** Fährt Trajektorie mit leerer Hand stur bis zum oberen Endpunkt ab.

