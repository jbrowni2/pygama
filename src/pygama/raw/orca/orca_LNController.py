"""
Decoder for ORCA Iseg HV
Written by James Browning 8/16/2022
"""

import logging
from typing import Any

from pygama.raw.orca.orca_header import OrcaHeader
from pygama.raw.orca.orca_packet import OrcaPacket
from pygama.raw.raw_buffer import RawBufferLibrary

from .orca_base import OrcaDecoder, get_ccc

log = logging.getLogger(__name__)


def calculate_mantissa(bin_list):
    mantissa = 1
    for i in range(0, len(bin_list)):
        if bin_list[i] == 1:
            mantissa += 2.0 ** (-(i + 2))

    return mantissa


class ORAmi286DecoderForLevel(OrcaDecoder):
    """Decoder for iSeg HV data written by ORCA."""

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:

        # store an entry for every event
        self.decoded_values_template = {
            "packet_id": {
                "dtype": "uint32",
            },
            "timestamp": {
                "dtype": "uint64",
                "units": "clock_ticks",
            },
            "crate": {
                "dtype": "uint64",
            },
            "card": {
                "dtype": "uint64",
            },
            "fillState": {
                "dtype": "uint64",
            },
            "lnLevel": {
                "dtype": "uint64",
            },
            "channel": {
                "dtype": "uint8",
            },
        }

        self.decoded_values = {}
        self.skipped_channels = {}
        super().__init__(
            header=header, **kwargs
        )  # also initializes the garbage df (whatever that means...)

    def set_header(self, header: OrcaHeader) -> None:
        self.header = header
        import copy

        self.decoded_values = copy.deepcopy(self.decoded_values_template)
        # for info in self.header:
        #    print(self.header[info])

        for card_dict in self.header["dataDescription"]:
            if card_dict == "Ami286Model":
                crate = 2
                card = 0
                for channel in range(0, 4):
                    ccc = get_ccc(crate, card, channel)
                    self.decoded_values[ccc] = copy.deepcopy(
                        self.decoded_values_template
                    )

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
        """Decode the ORCA Iseg HV packet."""
        """
        The packet is formatted as
        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
        ^^^^ ^^^^ ^^^^ ^^-----------------------data id
                         ^^ ^^^^ ^^^^ ^^^^ ^^^^-length in longs

        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
        ^^^^------------------------------------ fill state for level 3
        -----^^^^------------------------------- fill state for level 2
        ----------^^^^-------------------------- fill state for level 1
        ---------------^^^^--------------------- fill state for level 0
        -------------------------^^^^ ^^^^ ^^^^- device id
        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  level chan 0 encoded as a float
        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  time level 0 taken in seconds since Jan 1, 1970
        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  level chan 1 encoded as a float
        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  time level 1 taken in seconds since Jan 1, 1970
        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  level chan 2 encoded as a float
        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  time level 2 taken in seconds since Jan 1, 1970
        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  level chan 3 encoded as a float
        xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  time level 3 taken in seconds since Jan 1, 1970

        """
        evt_rbkd = rbl.get_keyed_dict()
        key_list = evt_rbkd.keys()
        crate = 2
        card = 0
        num_of_ch = 4

        for i in range(0, int(num_of_ch)):

            # The values for each channel will always be sent so just need
            # to check if they want to record the values for that channel or not.
            channel = i
            chan_check = get_ccc(crate, card, channel)
            if chan_check not in key_list:
                continue

            ccc = get_ccc(crate, card, channel)
            tbl = evt_rbkd[ccc].lgdo
            ii = evt_rbkd[ccc].loc
            if i == 0:
                tbl["fillState"].nda[ii] = (packet[1] >> 16) & 0xF
            elif i == 1:
                tbl["fillState"].nda[ii] = (packet[1] >> 20) & 0xF
            elif i == 2:
                tbl["fillState"].nda[ii] = (packet[1] >> 24) & 0xF
            elif i == 3:
                tbl["fillState"].nda[ii] = (packet[1] >> 28) & 0xF

            tbl["crate"].nda[ii] = crate
            tbl["card"].nda[ii] = card
            tbl["channel"].nda[ii] = channel
            tbl["timestamp"].nda[ii] = packet[3 + 2 * i]

            # The values being decoded are floats so we need to use IEEE 754 notation
            # First Calculate the LN Level
            sign = (packet[2 + i * 2] >> 31) & 0x1
            exponent = float((packet[2 + i * 2] >> 23) & 0xFF)
            bin_list = [int(d) for d in str(bin(packet[2 + i * 2]))[2:]]
            mantissa = calculate_mantissa(bin_list)
            level = ((-1) ** sign) * (2.0 ** (exponent - 127.0)) * (1 * mantissa)
            tbl["lnLevel"].nda[ii] = level

            evt_rbkd[ccc].loc += 1

        return evt_rbkd[ccc].is_full()
