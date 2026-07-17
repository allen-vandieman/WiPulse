from dataclasses import dataclass

@dataclass
class CSIConfig:

    bandwidth = 80

    chip = "43455c0"

    udp_port = 5500

    magic = b"\x11\x11"

    export_numpy = True

    export_csv = True