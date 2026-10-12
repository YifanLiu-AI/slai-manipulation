# Third-Party Dependencies

`wuji-retargeting` is vendored as a deliberately small runtime subset so the
simulation chain does not depend on another mutable checkout. The retained files
are its MIT license, package metadata, core retargeting package, RealSense MediaPipe
input adapter, and the single production configuration used by this project.

Upstream: `https://github.com/wuji-technology/wuji-retargeting.git`

Pinned commit: `6eafdb22085f0e29c1d58c62f88f77ae1e971d8c`

The upstream Git history, CI, tuning UI, documentation, datasets, and unrelated
examples are intentionally excluded. Keep project-specific patches in
`third_party/patches/` and record their purpose before adding them.

| Dependency | Expected location | Used by |
| --- | --- | --- |
| Isaac Lab | `/home/user/opt/IsaacLab` | UR5/Wuji simulation |
| wuji-retargeting | `third_party/wuji-retargeting` | MediaPipe retargeting |
| libspnav / spacenavd | System installation | SpaceMouse input |
| ur-rtde | `ur_env` Python environment | Real UR5 control |
| wuji-sdk / wujihandpy | Project Wuji virtual environment | Real Wuji control |

All dependencies retain their own licenses. Code outside the pinned retargeting
subset is consumed from the listed environment or system installation.
