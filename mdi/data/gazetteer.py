"""Static lookup tables for the dataset factory.

Coordinates are approximate and only need to be plausible: each landmark has a
seaward bearing range so sampled distress positions fall in open water.
"""

from __future__ import annotations

from dataclasses import dataclass

from mdi.schema import VesselType


@dataclass(frozen=True)
class Landmark:
    name: str
    lat: float
    lon: float
    seaward_bearing: tuple[int, int]  # degrees true, inclusive range
    rescue_centre: str


LANDMARKS: list[Landmark] = [
    Landmark("Veraval", 20.90, 70.36, (190, 280), "MRCC Mumbai"),
    Landmark("Mumbai Prongs Reef", 18.88, 72.80, (200, 300), "MRCC Mumbai"),
    Landmark("Aguada lighthouse", 15.49, 73.77, (200, 300), "MRCC Mumbai"),
    Landmark("Mangaluru", 12.87, 74.84, (200, 300), "MRSC Kochi"),
    Landmark("Kozhikode", 11.25, 75.77, (200, 300), "MRSC Kochi"),
    Landmark("Kochi lighthouse", 9.97, 76.24, (200, 300), "MRSC Kochi"),
    Landmark("Kollam", 8.88, 76.57, (200, 290), "MRSC Kochi"),
    Landmark("Vizhinjam", 8.38, 76.99, (190, 280), "MRSC Kochi"),
    Landmark("Kanyakumari", 8.08, 77.55, (150, 220), "MRCC Chennai"),
    Landmark("Thoothukudi", 8.80, 78.16, (60, 150), "MRCC Chennai"),
    Landmark("Rameswaram", 9.29, 79.31, (60, 120), "MRCC Chennai"),
    Landmark("Nagapattinam", 10.77, 79.85, (60, 120), "MRCC Chennai"),
    Landmark("Puducherry", 11.93, 79.84, (60, 120), "MRCC Chennai"),
    Landmark("Chennai lighthouse", 13.08, 80.29, (60, 120), "MRCC Chennai"),
    Landmark("Visakhapatnam", 17.69, 83.30, (90, 170), "MRCC Chennai"),
    Landmark("Paradip", 20.26, 86.67, (90, 170), "MRCC Chennai"),
]

# Vessel names are partitioned into train/test pools (see generator) so the NER
# model is evaluated on names it never saw during training.
VESSEL_NAMES: list[str] = [
    "Sri Murugan", "Matha", "Annai Velankanni", "St Antony", "Jai Hind",
    "Sagar Kanya", "Lakshmi Prasad", "Ganga Devi", "Kadal Rani", "Mayil",
    "Sea Princess", "Ocean Pearl", "Blue Dolphin", "Golden Wave", "Star of Kerala",
    "Al Ameen", "Bismillah", "Jesus Love", "Karunamayi", "Arabikadal",
    "Sea Hawk", "Mercy", "Vettaikaran", "Bharat Jyoti", "Coral Queen",
    "Malabar Star", "Thiruvalluvar", "Ayyappa", "Mahalakshmi", "Neptune Grace",
    "Konkan Kanya", "Saraswati", "Deep Sea Explorer", "Vasco", "Little Flower",
    "Sea Lord", "Kaveri", "Pearl of Tuticorin", "Silver Fin", "Pushpak",
    "Royal Mariner", "Nandini", "Matsya Shakti", "Morning Star", "Amma",
    "Indus Glory", "Bay Runner", "Southern Cross", "Thangam", "Madurai Veeran",
    "Durga Mata", "Sea Breeze", "Fathima", "Holy Cross", "Vellai Puravu",
    "Kanyakumari Amman", "Ocean Star", "Lady of Snows", "Rahmath", "Seaflower",
    "Karthika", "Hanuman Shakti", "Blue Marlin", "Sundari", "Jalaram",
    "Pride of Malabar", "Annapoorna", "Gift of God", "Sea Eagle", "Manjula",
    "Velankanni Matha", "Dolphin King", "Nila", "Sahyadri", "Captain Ram",
    "Kadal Pura", "Sri Balaji", "Rose Mary", "Silver Moon", "Ponnu",
    "Mother Mary", "Tiger Shark", "Arjun", "Sea Queen", "Gangamma",
    "Coral Reef", "Shanthi", "Neelakantan", "Abdul Kalam", "Lotus",
    "Island Hopper", "Navarathna", "Wave Rider", "Marine Hope", "Thamarai",
]

VESSEL_TYPE_WEIGHTS: dict[VesselType, float] = {
    VesselType.FISHING_BOAT: 0.40,
    VesselType.TRAWLER: 0.25,
    VesselType.CARGO: 0.10,
    VesselType.TANKER: 0.05,
    VesselType.FERRY: 0.07,
    VesselType.YACHT: 0.05,
    VesselType.TUG: 0.08,
}

# (min, max) persons on board per vessel type.
POB_RANGE: dict[VesselType, tuple[int, int]] = {
    VesselType.FISHING_BOAT: (2, 8),
    VesselType.TRAWLER: (5, 18),
    VesselType.CARGO: (12, 25),
    VesselType.TANKER: (15, 30),
    VesselType.FERRY: (20, 150),
    VesselType.YACHT: (1, 6),
    VesselType.TUG: (3, 8),
}

# Small craft rarely carry AIS/MMSI.
HAS_MMSI_PROB: dict[VesselType, float] = {
    VesselType.FISHING_BOAT: 0.15,
    VesselType.TRAWLER: 0.55,
    VesselType.CARGO: 1.0,
    VesselType.TANKER: 1.0,
    VesselType.FERRY: 0.95,
    VesselType.YACHT: 0.6,
    VesselType.TUG: 0.9,
}

INDIA_MID = "419"  # Maritime Identification Digits for India
