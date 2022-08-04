import logging
from typing import Any

import numpy as np

from pygama.raw.orca.orca_header import OrcaHeader
from pygama.raw.orca.orca_packet import OrcaPacket
from pygama.raw.raw_buffer import RawBufferLibrary

from .orca_base import OrcaDecoder, get_ccc

log = logging.getLogger(__name__)


class ORSIS3316WaveformDecoder(OrcaDecoder):
    """Decoder for SIS3316 ADC data written by ORCA."""

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:

        # store an entry for every event
        self.decoded_values_template = {
            "packet_id": {
                "dtype": "uint32",
            },
            "energy": {
                "dtype": "uint32",
                "units": "adc",
            },
            "energy_first": {
                "dtype": "uint32",
            },
            "timestamp": {
                "dtype": "uint64",
                "units": "clock_ticks",
            },
            "crate": {
                "dtype": "uint8",
            },
            "card": {
                "dtype": "uint8",
            },
            "channel": {
                "dtype": "uint8",
            },
            "waveform": {
                "dtype": "uint16",
                "datatype": "waveform",
                "wf_len": 65532,  # max value. override this before initalizing buffers to save RAM
                "dt": 8,  # override if a different clock rate is use
                "dt_units": "ns",
                "t0_units": "ns",
            },
        }

        # self.event_header_length = 1 #?
        self.decoded_values = {}
        self.skipped_channels = {}
        super().__init__(
            header=header, **kwargs
        )  # also initializes the garbage df (whatever that means...)

    def set_header(self, header: OrcaHeader) -> None:
        self.header = header
        import copy

        self.decoded_values = copy.deepcopy(self.decoded_values_template)

        for card_dict in self.header["ObjectInfo"]["Crates"][0]["Cards"]:
            if card_dict["Class Name"] == "ORSIS3316Model":
                card = card_dict["Card"]
                crate = 0
                for channel in range(0, 16):
                    ccc = get_ccc(crate, card, channel)
                    trace_length = card_dict["rawDataBufferLen"]
                    self.decoded_values[ccc] = copy.deepcopy(
                        self.decoded_values_template
                    )

                    if trace_length <= 0 or trace_length > 2**16:
                        raise RuntimeError(
                            "invalid trace_length: ",
                            trace_length,
                        )

                    self.decoded_values[ccc]["waveform"]["wf_len"] = trace_length

        self.decoded_values["waveform"]["wf_len"] = trace_length

    def get_key_list(self) -> list[int]:
        key_list = []
        for key in self.decoded_values.keys():
            key_list += [key]
        return key_list

    def get_decoded_values(self, key: int = None) -> dict[str, Any]:
        if key is None:
            dec_vals_list = self.decoded_values
            if len(dec_vals_list) == 0:
                raise RuntimeError("decoded_values not built yet!")
                return None
            return dec_vals_list  # Get first thing we find

        if key in self.decoded_values:
            dec_vals_list = self.decoded_values[key]
            return dec_vals_list
        raise RuntimeError("No decoded values for key", key)
        return None

    def decode_packet(
        self, packet: OrcaPacket, packet_id: int, rbl: RawBufferLibrary
    ) -> bool:
        """Decode the ORCA FlashCam ADC packet."""
        evt_rbkd = rbl.get_keyed_dict()

        evt_data_16 = np.frombuffer(packet, dtype=np.uint16)

        crate = (packet[1] >> 21) & 0xF
        card = (packet[1] >> 16) & 0x1F
        channel = (packet[1] >> 8) & 0xFF
        ccc = get_ccc(crate, card, channel)

        if ccc not in evt_rbkd:
            if ccc not in self.skipped_channels:
                self.skipped_channels[ccc] = 0
                log.debug(f"Skipping channel: {ccc}")
                log.debug(f"evt_rbkd: {evt_rbkd.keys()}")
            self.skipped_channels[ccc] += 1
            return False
        tbl = evt_rbkd[ccc].lgdo
        ii = evt_rbkd[ccc].loc

        tbl["crate"].nda[ii] = crate
        tbl["card"].nda[ii] = card
        tbl["channel"].nda[ii] = channel

        if len(packet) > 10:
            tbl["timestamp"].nda[ii] = packet[11] + ((packet[10] & 0xFFFF0000) << 16)
        else:
            tbl["timestamp"].nda[ii] = 0

        orca_helper_length16 = 54
        header_length16 = orca_helper_length16

        expected_wf_length = len(evt_data_16) - header_length16

        i_wf_start = header_length16

        i_wf_stop = i_wf_start + expected_wf_length

        if expected_wf_length > 0:

            if expected_wf_length == len(tbl["waveform"]["values"].nda[ii]):
                tbl["waveform"]["values"].nda[ii] = evt_data_16[i_wf_start:i_wf_stop]
            else:
                # this is else is to collected data that is doubled.
                # sometimes two events come in the same packet from ORCA.
                i_wf_stop = i_wf_start + len(tbl["waveform"]["values"].nda[ii])
                tbl["waveform"]["values"].nda[ii] = evt_data_16[i_wf_start:i_wf_stop]

                if ii != 8191:
                    evt_rbkd[ccc].loc += 1

                    ii = evt_rbkd[ccc].loc
                    offset = 5027

                    tbl["crate"].nda[ii] = (packet[offset + 1] >> 21) & 0xF
                    tbl["card"].nda[ii] = (packet[offset + 2] >> 16) & 0x1
                    tbl["channel"].nda[ii] = 33
                    tbl["timestamp"].nda[ii] = packet[offset + 11] + (
                        (packet[offset + 10] & 0xFFFF0000) << 16
                    )

                    i_wf_start = len(tbl["waveform"]["values"].nda[ii]) + 54
                    i_wf_stop = i_wf_start + len(tbl["waveform"]["values"].nda[ii])
                    tbl["waveform"]["values"].nda[ii] = evt_data_16[
                        i_wf_start:i_wf_stop
                    ]

        evt_rbkd[ccc].loc += 1

        return evt_rbkd[ccc].is_full()
