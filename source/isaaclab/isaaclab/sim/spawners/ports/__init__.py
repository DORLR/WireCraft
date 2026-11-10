"""Sub-module for spawning custom procedural port objects.
"""

from .male_port_spawner import ProceduralMalePortCfg, spawn_procedural_male_port
from .female_port_spawner import ProceduralFemalePortCfg, spawn_procedural_female_port
from .clip_spawner import ProceduralClipCfg, spawn_procedural_clip

from .connector_profiles import ConnectorProfile, connector_families, get_connector_profile
from .connector_spawner import PortCfg, SocketCfg, spawn_connector_port, spawn_connector_socket
