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


class ORiSegHVCardDecoderForHV(OrcaDecoder):
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
            "current": {
                "dtype": "uint64",
            },
            "voltage": {
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

        for card_dict in self.header["ObjectInfo"]["Crates"][1]["Cards"]:
            card = card_dict["Card"]
            if card_dict["Class Name"] == "OREHS8260pModel":
                crate = 1
                for channel in range(0, len(card_dict["targets"])):
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
         ^^^^ ^^^^ ^^^^ ^^----------------------- Data ID (from header)
         -----------------^^ ^^^^ ^^^^ ^^^^ ^^^^- length
         xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
         --------^-^^^--------------------------- Crate number
         -------------^-^^^^--------------------- Card number
         --------------------^^^^ ^^^^----------- Chan number
         xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  spare
         xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  spare
         xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  unix Time
         xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  Chan 0 Actual Voltage as Float
         xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx  Chan 0 Actual Current as Float
         Repeated for number of channels
"""
        evt_rbkd = rbl.get_keyed_dict()
        key_list = evt_rbkd.keys()
        crate = 1
        card = (packet[1] >> 16) & 0x1F
        num_of_ch = (packet[1] >> 4) & 0xF
        timestamp = packet[4]

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

            tbl["crate"].nda[ii] = crate
            tbl["card"].nda[ii] = card
            tbl["channel"].nda[ii] = channel
            tbl["timestamp"].nda[ii] = timestamp

            # The values being decoded are floats so we need to use IEEE 754 notation
            # First Calculate the Voltage
            sign = (packet[5 + i * 2] >> 31) & 0x1
            exponent = float((packet[5 + i * 2] >> 23) & 0xFF)
            bin_list = [int(d) for d in str(bin(packet[5 + i * 2]))[2:]]
            mantissa = calculate_mantissa(bin_list)
            volt = ((-1) ** sign) * (2.0 ** (exponent - 127.0)) * (1 * mantissa)
            tbl["voltage"].nda[ii] = volt

            # Next Calculate the Current
            sign = (packet[6 + i * 2] >> 31) & 0x1
            exponent = float((packet[6 + i * 2] >> 23) & 0xFF)
            bin_list = [int(d) for d in str(bin(packet[6 + i * 2]))[2:]]
            mantissa = calculate_mantissa(bin_list)
            current = ((-1) ** sign) * (2.0 ** (exponent - 127.0)) * (1 * mantissa)
            tbl["current"].nda[ii] = current
            evt_rbkd[ccc].loc += 1

        return evt_rbkd[ccc].is_full()
