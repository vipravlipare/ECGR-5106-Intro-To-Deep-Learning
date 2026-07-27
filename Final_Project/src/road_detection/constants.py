from __future__ import annotations

PROJECT_CLASSES = [
    "car",
    "bus",
    "truck",
    "pedestrian",
    "traffic light",
    "traffic sign",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(PROJECT_CLASSES)}

BDD_ALIASES = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}

