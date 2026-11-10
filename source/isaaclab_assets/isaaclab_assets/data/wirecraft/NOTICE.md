# WireCraft assets

- `connectors/printable/`: cylinder, cuboid, stacked cuboid, and DisplayPort-style port/socket pairs.
- `connectors/real/`: Ethernet, DisplayPort, HDMI, and USB-A visual and reference CAD pairs.
- `connectors/real/*/holder/`: printable socket mounts.
- `fixtures/`: printable clips, straight/curved channels, and backboard.

STEP files use millimeters; USD visuals use meters. Printable ports include a rear cable glue hole: `port.step` is 8 mm in diameter, and `port_wire6.step` is 6 mm.

The port origin is its tip's rear shoulder; the socket origin is its mouth. Local +X is the insertion direction. Simulation uses simplified collision geometry configured in `connector_profiles.py`, `PortCfg`, and `SocketCfg`; contacts and socket tongues are visual-only. Runtime overrides do not change the packaged CAD or visuals.

Real connector STEP files are references, not printing files. USB-A STEP files are equivalent geometry reconstructed from the simulation profile. DisplayPort `port.step` has invalid B-rep geometry, and HDMI `socket.step` contains surfaces only; neither is suitable for manufacturing.

## Robots

- `robots/ur5_with_gripper_camera_v2.usd`: UR5 with a Robotiq 2F-85 gripper and an Intel RealSense D455 wrist
  camera, assembled for WireCraft. Copied from `WireManip/ur5-wire-asset`, last changed in commit `d082980`.
- `robots/table.usd`: the workstation table, from the same commit.
- Franka uses Isaac Lab's official `FrankaEmika/panda_instanceable.usd` from Nucleus, and Trossen uses
  `assets/robots/stationary_ai/stationary_ai.usd` from `TrossenRobotics/trossen_ai_isaac` at commit `01d36c6`,
  loaded from GitHub. Neither is copied here. WireCraft adds a `grasp_center` body to both at load time.

`WIRECRAFT_ROBOT_ASSET_DIR` replaces this `robots/` folder.
