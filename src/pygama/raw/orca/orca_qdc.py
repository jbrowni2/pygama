"""
Decoder for ORCA ORSIS3316
Written by James Browning 8/16/2022
"""

import logging
from typing import Any

from pygama.raw.orca.orca_header import OrcaHeader
from pygama.raw.orca.orca_packet import OrcaPacket
from pygama.raw.raw_buffer import RawBufferLibrary

from .orca_base import OrcaDecoder, get_ccc

log = logging.getLogger(__name__)


class ORCAEN792NDecoderForQdc(OrcaDecoder):
    """Decoder for iSeg HV data written by ORCA."""

    def __init__(self, header: OrcaHeader = None, **kwargs) -> None:

        # store an entry for every event
        self.decoded_values_template = {
            "packet_id": {
                "dtype": "uint32",
            },
            "timestamp": {
                "dtype": "uint64",
            },
            "timestampMicro": {
                "dtype": "uint64",
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
            "energy": {
                "dtype": "uint64",
            },
            "eventCounter": {
                "dtype": "uint64",
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

        for card_dict in self.header["ObjectInfo"]["Crates"][0]["Cards"]:
            card = card_dict["Card"]
            if card_dict["Class Name"] == "ORCaen792Model":
                crate = 0
                for channel in range(0, 16):
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
         -----------------^^ ^^^^ ^^^^ ^^^^ ^^^^- n-longs+2
         xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
         ----------^^^^-------------------------- Crate number
         ---------------^^^^--------------------- Card number
         n-longs of data described in documentation.


"""

        evt_rbkd = rbl.get_keyed_dict()
        key_list = evt_rbkd.keys()
        length = packet[0] & 0x3FFFF
        crate = (packet[1] >> 20) & 0xF
        card = (packet[1] >> 16) & 0xF
        time_check = packet[1] & 0x1
        if time_check == 1:
            data_start = 4
            timestamp = packet[2]
            time_micro = packet[3]
        else:
            data_start = 2
            timestamp = -1
            time_micro = -1

        for i in range(data_start, length - 1):
            # if i == dataStart:
            #    print("first")
            channel = (packet[i] >> 17) & 0x3F
            chan_check = get_ccc(crate, card, channel)
            if chan_check not in key_list:
                continue

            ccc = get_ccc(crate, card, channel)
            tbl = evt_rbkd[ccc].lgdo
            ii = evt_rbkd[ccc].loc

            tbl["crate"].nda[ii] = crate
            tbl["card"].nda[ii] = card
            tbl["channel"].nda[ii] = channel
            tbl["energy"].nda[ii] = packet[i] & 0xFFF
            tbl["eventCounter"].nda[ii] = packet[length - 1]
            tbl["timestamp"].nda[ii] = timestamp
            tbl["timestampMicro"].nda[ii] = time_micro
            evt_rbkd[ccc].loc += 1

        return evt_rbkd[ccc].is_full()
