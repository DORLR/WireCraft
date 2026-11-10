# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-independent wire backends, connector assembly and shared wire semantics."""

from .connectors import ConnectorCfg, ConnectorGeometry, normalize_family
from .wire_cfg import WireSceneCfg
from .wire_state import FarAnchor, WireState
