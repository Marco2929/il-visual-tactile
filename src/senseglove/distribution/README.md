# SenseGlove Distribution - Lab Setup (IPA-PC-327RFA5)

Lab-specific notes for this workcell. For vendor setup instructions (Docker
install, image pull, token) see [`INTEGRATION_README.md`](INTEGRATION_README.md).

> **Our IP differs from the vendor docs.** The vendor README documents the
> factory link-local addresses `169.254.186.72` (right) / `.73` (left).
> Our DG5F-R is configured as **`192.168.56.102`**, so `--delto-ip` must be
> passed on every launch or the driver will time out on the wrong address.

---

## TL;DR

One-time, after every image pull:

```bash
bash build_cyclone_image.sh          # creates :cyclone-latest from :latest
```

Then two terminals, glove first:

```bash
# Terminal 1 - glove
uv run start.py r1 --tag cyclone-latest --num-gloves 1

# Terminal 2 - hand (read the glove ID from terminal 1 first!)
uv run start.py tesollo --tag cyclone-latest \
  --hand right --glove-id <ID> --delto-ip 192.168.56.102
```

`--tag cyclone-latest` is what makes the SenseGlove topics visible to the
rest of the workspace. Without it both containers run Fast DDS and are
invisible to `ros2_dev` - see [DDS / Cyclone overlay](#dds--cyclone-overlay).

---

## Required arguments, and why

| Flag | Value | Why it is mandatory here |
|---|---|---|
| `--tag` | `cyclone-latest` | The stock image ships **Fast DDS only**, which does not interoperate with the Cyclone DDS used by every other container. Without this tag the stack runs fine in isolation but no topic crosses the container boundary. |
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

## DDS / Cyclone overlay

The rest of this workspace (`ros2_dev`, `zed_container`, `ur_artus`) runs
**Cyclone DDS** pinned to loopback via `docker/cyclonedds.xml`. The prebuilt
SenseGlove image ships **Fast DDS** only:

```
senseglove_tesollo:  librmw_cyclonedds_cpp.so  ->  NOT FOUND
                     librmw_fastrtps_cpp.so    ->  present
```

The two implementations do not interoperate, so DG5F joint states never reach
the recorder. `Dockerfile.cyclone` adds `ros-jazzy-rmw-cyclonedds-cpp` as a
thin layer on top of the pulled image; the compose files then set
`RMW_IMPLEMENTATION` + `CYCLONEDDS_URI` and mount the shared XML.

### Build the overlay

```bash
bash build_cyclone_image.sh                      # :latest     -> :cyclone-latest
BASE_TAG=tesollo-1.2.0 bash build_cyclone_image.sh   # -> :cyclone-tesollo-1.2.0
```

> **Rebuild after every `uv run image_manager.py --pull`.** The overlay is
> pinned to the base image it was built from; a fresh pull leaves it stale.

### Both containers must match

The glove path is entirely internal to this stack:

```
senseglove_r1  --/r1/glove70/rh/glove_states-->  mapper in senseglove_tesollo
```

So switching only one side breaks **teleop itself**, not just recording - the
mapper goes deaf and the hand stops following the glove. Always start both with
the same `--tag`.

### Reverting without editing files

Both compose files read `${SENSEGLOVE_RMW:-rmw_cyclonedds_cpp}`. To go back to
the stock image and old behaviour:

```bash
SENSEGLOVE_RMW=rmw_fastrtps_cpp uv run start.py tesollo --tag latest --hand right --glove-id 70
```

If you start the **stock** image without that override, Cyclone cannot be
loaded and the nodes die immediately:

```
failed to load shared library 'librmw_cyclonedds_cpp.so'
```

That is the symptom of a missing/stale `cyclone-*` build.

---

## MoveIt and the `/joint_states` relay

MoveIt's planning scene monitor subscribes **only to `/joint_states`**, but the
hand publishes on `/dg5f_right/joint_states`. If nothing bridges the two,
`move_group` and `servo_node` warn forever:

```
[move_group] The complete state of the robot is not yet known.
  Missing rj_dg_1_1, rj_dg_1_2, ... rj_dg_5_4
```

Note the joint **names already match** the URDF - this is a topic/namespace
issue, not a naming one, so it is easy to misread as a DDS failure. The topics
are visible from `ros2_dev`; they are just on the wrong topic for MoveIt.

This used to be covered by `tesollo_control` inside
`ur_tesollo_moveit.launch.py`, but that is commented out - it includes
`dg5f_driver`, which would contend with this container for Modbus 502 (see
[Only one Modbus client at a time](#only-one-modbus-client-at-a-time)).

So `start.py tesollo` now prepends a relay:

```
ros2 run topic_tools relay /dg5f_right/joint_states /joint_states &
exec ros2 launch dg5f_interaction dg5f_robot_hand_mapper.launch.py ...
```

The monitor merges publishers by joint name, so the UR's 6 joints and the
hand's 20 coexist on `/joint_states`. Confirm with:

```bash
docker exec ros2_dev bash -c 'ros2 topic echo /joint_states --once | head -30'
```

You should see both `shoulder_pan_joint`-style and `rj_dg_*` names arriving.

If you re-enable `tesollo_control`, disable the relay to avoid two publishers
fighting over the same joints:

```bash
uv run start.py tesollo --tag cyclone-latest --hand right --glove-id 70 --no-joint-state-relay
```

> The relay lives in `build_launch_command` in `start.py`, **not** in the
> compose file. `run_compose` overwrites the compose `command:` wholesale, so
> edits there are silently discarded.

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
# NOTE: use docker exec, not `uv run start.py bash` - see caveat below.
docker exec -it senseglove_tesollo bash
ros2 topic hz /r1/glove70/rh/glove_states                       # glove -> mapper
ros2 topic echo /dg5f_right/dg5f_right_controller/joint_trajectory --once
```

And that the topics are visible from the *other* containers - this is the whole
point of the Cyclone overlay:

```bash
docker exec ros2_dev bash -c 'ros2 topic list | grep dg5f'
docker exec ros2_dev bash -c 'ros2 topic hz /dg5f_right/joint_states'
```

> **`uv run start.py bash` does not inherit the DDS config.** That subcommand
> shells out to a raw `docker run`, bypassing compose, so it gets neither
> `RMW_IMPLEMENTATION` nor the `cyclonedds.xml` mount and will see an empty
> topic list. Either `docker exec` into a running container as above, or pass
> the settings by hand:
>
> ```bash
> docker run --rm -it --network host --privileged \
>   -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
>   -e CYCLONEDDS_URI=file:///tmp/cyclonedds.xml \
>   -v "$PWD/../../docker/cyclonedds.xml:/tmp/cyclonedds.xml:ro" \
>   zippeth/senseglove_robotics:cyclone-latest bash
> ```

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

Moving the SenseGlove stack onto Cyclone **adds to this budget**: both
`senseglove_r1` and `senseglove_tesollo` (ros2_control + 7 controllers + mapper
+ spawners) now claim slots that previously sat on a separate Fast DDS domain.
If you hit the error above, count what is actually registered before raising
the limit further:

```bash
docker exec ros2_dev bash -c 'ros2 node list | wc -l'
```

---

## Full flag reference

```bash
uv run start.py --help
uv run start.py tesollo --hand right --glove-id 70 \
  --delto-ip 192.168.56.102 --no-run      # print the command without running
```

`--no-run` is the quickest way to confirm which launch arguments are forwarded.