# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named wire presets for common use cases."""

from .wire_cfg import WireCfg, WirePortCfg


class WirePresets:
    """Collection of named wire configurations for common use cases."""

    THIN_FLEXIBLE = WireCfg(
        num_segments=20,
        total_length=0.3,
        radius=0.001,
        stiffness=0.5,
        damping=0.05,
    )
    """Thin flexible wire — 20 segments, 30cm, 1mm radius, low stiffness."""

    THICK_RIGID = WireCfg(
        num_segments=10,
        total_length=0.2,
        radius=0.005,
        stiffness=50.0,
        damping=5.0,
    )
    """Thick rigid wire — 10 segments, 20cm, 5mm radius, high stiffness."""

    MEDIUM_CABLE = WireCfg(
        num_segments=15,
        total_length=0.4,
        radius=0.003,
        stiffness=5.0,
        damping=0.5,
    )
    """Medium cable — 15 segments, 40cm, 3mm radius, moderate stiffness."""

    USB_A_CABLE = WireCfg(
        num_segments=12,
        total_length=0.4,
        radius=0.004,
        stiffness=5.0,
        damping=1.0,
        joint_limit=0.5,
        fix_base=True,
        density=2000.0,
        self_collision=True,
        # color=(0.8, 0.2, 0.1),
        color=(0.0, 0.0, 0.0),
        start_port=WirePortCfg(family="usb_a"),
        end_port=WirePortCfg(family="usb_a"),
    )
    """Cable with shared USB-A connector profiles attached at both ends."""
