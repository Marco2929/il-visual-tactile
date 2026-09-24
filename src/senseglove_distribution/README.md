# SenseGlove Distribution - Lab Setup (IPA-PC-327RFA5)

Lab-specific notes for this workcell. For vendor setup instructions (Docker
install, image pull, token) see [`INTEGRATION_README.md`](INTEGRATION_README.md).

> **Our IP differs from the vendor docs.** The vendor README documents the
> factory link-local addresses `169.254.186.72` (right) / `.73` (left).
> Our DG5F-R is configured as **`192.168.56.102`**, so `--delto-ip` must be
> passed on every launch or the driver will time out on the wrong address.

---

## TL;DR

Two terminals, glove first:

```bash
# Terminal 1 - glove
uv run start.py r1 --num-gloves 1

# Terminal 2 - hand (read the glove ID from terminal 1 first!)
uv run start.py tesollo --hand right --glove-id <ID> --delto-ip 192.168.56.102
```

---

## Required arguments, and why

| Flag | Value | Why it is mandatory here |
|---|---|---|
| `--hand` | `right` | Without it the launcher defaults to the **left** hand: it starts `DG5FLeftHardware` and connects to `169.254.186.73`. It also gates the `delto_ip_rh:=` argument that the mapper actually reads. |
| `--delto-ip` | `192.168.56.102` | Image default is `169.254.186.72`. |
| `--glove-id` | glove serial | Builds the topic name. A wrong ID means the mapper subscribes to a topic nobody publishes - everything starts "successfully" and the hand never moves. |

### Finding the glove ID

`start.py r1` prints it on connect:

```
SG_API: Device connected!    ID:70 | Connected device_ids:[70]
```

That `ID:70` is the value for `--glove-id`. Cross-check that the mapper agrees -
it logs the topic it subscribes to:

```
[dg5f_robot_hand_mapper]: Subscribing to: /r1/glove70/rh/glove_states
```

If that says `glove51` (the launch-file default) while your glove is `70`, the
ID was not passed through.

---

## Network

The hand shares the `192.168.56.0/24` subnet with the UR10e (`192.168.56.101`).
Both must be reachable from the same interface.

```bash
ping -c2 192.168.56.102          # hand
ip route get 192.168.56.102      # must NOT say "dev wlp6s0"
```

If the route goes via `wlp6s0`, no Ethernet interface holds a `192.168.56.x`
address and packets leak to the WiFi gateway. Bring the wired profile up:

```bash
nmcli device status
nmcli connection up netplan-NM-04bc8b3b-4538-3c7d-848d-6062e286c54d
```

That profile is bound to `enp5s0` and carries `192.168.56.100/24`.

---

## Only one Modbus client at a time

The DG5F accepts a **single** Modbus TCP connection on port 502. Two things in
this workspace both want it:

- `senseglove_tesollo` (this launcher)
- `ros2 launch il_main ur_tesollo_moveit.launch.py` (includes `dg5f_driver`)

Running both gives `Connect Failed to 192.168.56.102:502` in whichever started
second - misleading, because the hand is powered on and fine. Check and stop the
other owner:

```bash
ss -tnp | grep 192.168.56.102
docker ps --filter name=senseglove
docker stop senseglove_tesollo
```

A fast failure (<1 ms) means the hand actively refused the connection, i.e.
someone else holds it. A ~3 s failure is a SYN timeout, i.e. a routing/link
problem instead.

---

## Verifying a good start

A healthy Tesollo launch prints:

```
Configured Model: DG5F-R (0x5F22)   Status: OK (Model matched)
Successfully connected. Firmware: v3.0
Successfully switched controllers!        (x8: 5 fingertips, JSB, JTC)
```

Then check the data actually flows:

```bash
uv run start.py bash
ros2 topic hz /r1/glove70/rh/glove_states                       # glove -> mapper
ros2 topic echo /dg5f_right/dg5f_right_controller/joint_trajectory --once
```

---

## Force sensor calibration

The fingertip F/T sensors drift and need periodic re-zeroing:

```bash
uv run start.py bash
ros2 service call /dg5f_right/calibrate_force_sensors std_srvs/srv/Trigger
```

Do this with the fingers unloaded and not touching anything.

---

## Known rough edges

**Orphan-container warnings.** `start.py r1` and `start.py tesollo` share one
Compose project and each rewrite `compose_runtime.yml`, so starting the second
service makes the first look like an orphan:

```
WARN Found orphan containers ([senseglove_r1]) for this project
open .../compose_runtime.yml: no such file or directory
```

Harmless, but stopping one container can disturb the other. Start the glove
first and let it settle.

**DDS participant limit.** Our `docker/cyclonedds.xml` restricts discovery to
`lo` with multicast off, so every participant needs a numbered port slot. These
containers are `network_mode: host` and share that range with `ur_artus`. Too
many nodes gives:

```
Failed to find a free participant index for domain 0
```

`MaxAutoParticipantIndex` is raised to 60 in `docker/cyclonedds.xml` to allow
for this; processes must be restarted to pick up changes to that file.

---

## Full flag reference

```bash
uv run start.py --help
uv run start.py tesollo --hand right --glove-id 70 \
  --delto-ip 192.168.56.102 --no-run      # print the command without running
```

`--no-run` is the quickest way to confirm which launch arguments are forwarded.