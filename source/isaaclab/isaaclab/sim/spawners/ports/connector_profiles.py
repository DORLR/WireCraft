# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Port dimensions shared by both wire models. All lengths are in meters.

The port's +X axis points toward its leading face. Its origin is the tip's
rear shoulder. Socket +X points inward from the mouth. Real profiles use
simplified collision geometry.
"""

from dataclasses import dataclass
import math


def rectangle(y_min, y_max, z_min, z_max):
    """Return a counterclockwise cross-section in the local YZ plane."""
    return ((y_min, z_min), (y_max, z_min), (y_max, z_max), (y_min, z_max))


@dataclass(frozen=True)
class ConnectorProfile:
    family: str
    category: str
    tip_length: float
    tip_outline: tuple[tuple[float, float], ...]
    housing_bounds: tuple[float, float, float, float, float, float]
    socket_outer_bounds: tuple[float, float, float, float]
    socket_depth: float
    clearance: float
    grasp_x: float
    latch_bounds: tuple[float, float, float, float, float, float] | None = None
    housing_sections: tuple[tuple[float, float, float, float, float, float], ...] | None = None
    socket_mount_panel_size: tuple[float, float, float] | None = None
    """Optional mounting enclosure (Y size, Z size, minimum X extent), in meters."""
    latch_collision_height_scale: float = 1.0
    """Compressed key height fraction shared by port collision and socket relief."""

    @property
    def collision_latch_bounds(self) -> tuple[float, float, float, float, float, float] | None:
        """Effective fixed-key bounds shared by the port and matching socket."""
        if self.latch_bounds is None:
            return None
        scale = self.latch_collision_height_scale
        if not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("Latch collision height scale must be finite and in (0, 1].")
        *bounds, z0, z1 = self.latch_bounds
        return (*bounds, z0, z0 + (z1 - z0) * scale)

    def cable_anchor_x(self, embed_depth: float = 0.008) -> float:
        """Cable endpoint inside the housing, measured in the port frame."""
        if not 0 < embed_depth <= self.housing_bounds[1] - self.housing_bounds[0]:
            raise ValueError("Cable embed depth must fit inside the housing.")
        return self.housing_bounds[0] + embed_depth


_PRINT_HOUSING = (-0.015, 0.0, -0.009, 0.009, -0.005, 0.005)
_PRINT_SOCKET = (-0.020, 0.020, -0.020, 0.020)

_PROFILES = {
    "cylinder": ConnectorProfile(
        "cylinder", "printable", 0.015,
        tuple((0.005 * math.cos(i * math.tau / 16), 0.005 * math.sin(i * math.tau / 16)) for i in range(16)),
        _PRINT_HOUSING, _PRINT_SOCKET, 0.020, 0.003, -0.014,
    ),
    "cuboid": ConnectorProfile(
        "cuboid", "printable", 0.015, rectangle(-0.0075, 0.0075, -0.005, 0.005),
        _PRINT_HOUSING, _PRINT_SOCKET, 0.020, 0.003, -0.014,
    ),
    "stacked_cuboid": ConnectorProfile(
        "stacked_cuboid", "printable", 0.015, rectangle(-0.0075, 0.0075, -0.0045, 0.0045),
        _PRINT_HOUSING, _PRINT_SOCKET, 0.020, 0.003, -0.014,
        latch_bounds=(-0.006, 0.006, -0.0025, 0.0025, 0.005, 0.009),
    ),
    "displayport_style": ConnectorProfile(
        "displayport_style", "printable", 0.015,
        ((-0.008, -0.0025), (0.005, -0.0025), (0.008, -0.0005), (0.008, 0.0025), (-0.008, 0.0025)),
        _PRINT_HOUSING, _PRINT_SOCKET, 0.020, 0.003, -0.014,
    ),
    "ethernet": ConnectorProfile(
        "ethernet", "real", 0.013000001, rectangle(-0.00584, 0.00584, -0.002225, 0.004975),
        (-0.034, 0.0, -0.00584, 0.00584, -0.00553, 0.00553),
        (-0.0095, 0.0095, -0.0065, 0.013), 0.016, 0.00025, -0.020,

        latch_bounds=(-0.0045, 0.013000001, -0.00305, 0.00305, 0.004975, 0.010264),
        latch_collision_height_scale=0.5,
        housing_sections=(
            # Soft-boot Y/Z diameter: (8.60 + 13.52) / 2 = 11.06 mm.
            (-0.034, -0.0088, -0.00553, 0.00553, -0.00553, 0.00553),
            (-0.009441, 0.0, -0.00584, 0.00584, -0.003325, 0.004975),
        ),
    ),
    "usb_a": ConnectorProfile(
        "usb_a", "real", 0.011, rectangle(-0.00665, 0.00665, -0.003586, 0.002074),
        (-0.017, 0.0, -0.008, 0.008, -0.005, 0.005),
        (-0.0088, 0.0088, -0.0052, 0.0048), 0.014, 0.00025, -0.017,

    ),
    "displayport": ConnectorProfile(
        "displayport", "real", 0.0111,
        ((-0.00795, -0.00233), (0.00627, -0.00233), (0.00795, -0.00065),
         (0.00795, 0.00233), (-0.00795, 0.00233)),
        (-0.0365, 0.0, -0.01008512, 0.01008512, -0.00600335, 0.00694275),
        (-0.009, 0.009, -0.004, 0.004), 0.013, 0.00025, -0.022,
        socket_mount_panel_size=(0.024, 0.018, 0.015),
    ),
    "hdmi": ConnectorProfile(
        "hdmi", "real", 0.0125,
        ((-0.004425, -0.002225), (0.004425, -0.002225),
         (0.00695, -0.00046), (0.00695, 0.002225),
         (-0.00695, 0.002225), (-0.00695, -0.00046)),
        (-0.020, 0.0, -0.009, 0.009, -0.005, 0.005),
        (-0.008, 0.008, -0.0038, 0.0033), 0.01055, 0.00025, -0.012,
        socket_mount_panel_size=(0.022, 0.017, 0.01255),
    ),
}


def connector_families() -> tuple[str, ...]:
    return tuple(_PROFILES)


def get_connector_profile(family: str) -> ConnectorProfile:
    try:
        return _PROFILES[family]
    except KeyError:
        raise ValueError(f"Unknown connector family {family!r}; choose from {connector_families()}.") from None
