# DG5F Temperature Watchdog

Monitors real per-joint motor temperature on the Tesollo DG5F hand and
deactivates the controller before it overheats. Intended to run alongside
normal SenseGlove teleoperation.

## What to run

**Terminal 1 — glove**
```bash
cd ~/dev/il-visual-tactile/src/senseglove_distribution
uv run start.py r1 --num-gloves 1
```

**Terminal 2 — hand + mapper**
```bash
cd ~/dev/il-visual-tactile/src/senseglove_distribution
uv run start.py tesollo --hand right --glove-id 70
```

**Terminal 3 — watchdog** (start last, needs the stack up)
```bash
cd ~/dev/il-visual-tactile
bash senseglove_guard/run_watchdog.sh
```

That's the long-term run. Stop with `Ctrl-C`.

## Output

Both files land on the **host** in `senseglove_guard/logs/`, named by start time:

| File | Contents |
|------|----------|
| `watchdog_<stamp>.csv` | one row per joint per report: `elapsed_s, joint, temp_c, slope_c_per_min` |
| `watchdog_<stamp>.log` | the console output |

The CSV is synced out of the container every 5 s, so a crash or container
restart loses at most a few seconds.

## Thresholds

| | default | meaning |
|---|---|---|
| warn | 58 °C | logs a warning naming the joint |
| trip | 65 °C | deactivates `dg5f_right_controller` |
| vendor limit | 75 °C | per Tesollo, not used directly |

Idle sits around 45–50 °C, and measured rise under normal teleop is about
0.1 °C/min, so 65 °C leaves plenty of headroom plus 10 K of margin below the
vendor limit. The margin matters because the case-mounted sensor lags the
winding — the winding heats in seconds, the sensor follows minutes later.

## After a trip

The hand stops responding to the glove. Let it cool, then re-arm:

```bash
docker exec senseglove_tesollo bash -c 'source /ros_ws/install/setup.bash;
  ros2 control switch_controllers --activate dg5f_right_controller \\
    -c /dg5f_right/controller_manager'
```

Check state at any time:
```bash
docker exec senseglove_tesollo bash -c 'source /ros_ws/install/setup.bash;
  ros2 control list_controllers -c /dg5f_right/controller_manager'
```

## Useful options

```bash
bash senseglove_guard/run_watchdog.sh --table          # all 20 joints, live
bash senseglove_guard/run_watchdog.sh --stop-at 60     # lower trip point
bash senseglove_guard/run_watchdog.sh --dry-run        # warn only, never trip
HAND=left bash senseglove_guard/run_watchdog.sh        # left hand
```

For a long run leave `--table` off: it reprints the whole table each cycle and
bloats the log file. The default one-line summary is enough, and the CSV has
the per-joint detail.

Verify the trip path without heating anything — set the threshold below the
current temperature:
```bash
bash senseglove_guard/run_watchdog.sh --stop-at 40 --warn-at 38
```
This **really deactivates** the controller. Re-arm afterwards.

## How it works

`/dg5f_<hand>/dynamic_joint_states` carries a per-joint `temperature` field at
100 Hz. It comes from the `JointStateBroadcaster`, which exports every
interface the hardware component offers — which is why temperature is
available even though `dg5f_right_ros2_control.xacro` only declares `position`
and `velocity`.

On trip, the watchdog calls `switch_controller` to deactivate
`dg5f_right_controller`. That matters: the controller is effort-based
(`p: 1.5`), and per the driver source the firmware keeps applying the last PWM
duty it received — so merely killing the command stream would leave the hand
pressing and heating. `joint_state_broadcaster` stays active, so temperature
monitoring keeps working after a trip.

## Why not a separate container

Fast DDS uses shared memory for same-host traffic, which needs a shared IPC
namespace. `--network host` only shares the network namespace, and the
SenseGlove containers run `IpcMode=private`. A sidecar container therefore
receives nothing — mounting `/dev/shm` does not help, and `--ipc=container:`
is refused because the donor is not shareable. So the script `docker cp`s the
watchdog into the running teleop container and executes it there.

## Known limitations

- **Trip verified, physical release not.** A real trip was confirmed:
  controller went `active` → `inactive` in ~120 ms. But immediately afterwards
  `list_hardware_interfaces` still showed 20 effort interfaces as `[claimed]`.
  Whether the motors actually go slack is unconfirmed; if they do not, the
  watchdog should command the current position *before* deactivating.
- **Thresholds are not vendor-calibrated.** 58/65 °C are chosen against the
  75 °C figure and measured idle behaviour, not a datasheet curve.
- **Re-arm behaviour untested.** The controller may resume with a stale
  setpoint. Keep hands clear the first time.
- **Does not prevent self-collision.** Spreading ring and little finger can
  wedge them mechanically (observed once, required unplugging). The watchdog
  only reacts to the resulting heat, it cannot stop the wedging.

## Files

| File | Purpose |
|---|---|
| `dg5f_temp_watchdog.py` | the watchdog node |
| `run_watchdog.sh` | launcher, handles CSV sync to host |
| `logs/` | CSV + log output, plus earlier `cal_run*.csv` measurements |
