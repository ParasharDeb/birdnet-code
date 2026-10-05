"""Load BirdNET and report how each target species maps onto the model's labels.

Run this before trusting the API: a species that isn't in the model can never be
detected, no matter how good the recording is.

    python -m scripts.check_species
"""

from __future__ import annotations

import sys

from app.analyzer import service
from app.species import TARGET_SPECIES


def main() -> int:
    print("Loading BirdNET (the first run downloads the model)...\n")
    service.load()

    if not service.ready:
        print(f"FAILED to load model: {service.load_error}")
        return 2

    print(f"Model labels: {len(service.labels)}\n")
    width = max(len(s.common_name) for s in TARGET_SPECIES)
    missing = []

    for spec in TARGET_SPECIES:
        res = service.resolutions[spec.key]
        if service.scored_by(spec.key) == "custom":
            print(f"  OK      {spec.emoji} {spec.common_name:<{width}}  ->  custom classifier")
        elif res.resolved:
            print(
                f"  OK      {spec.emoji} {spec.common_name:<{width}}  ->  "
                f"{res.label}   (matched on {res.matched_on} name)"
            )
            for extra in res.extra_labels:
                print(f"  {'':<{width + 12}}+ {extra}")
        else:
            missing.append(spec)
            print(f"  MISSING {spec.emoji} {spec.common_name:<{width}}  ->  not in this model")

    print()
    if missing:
        print(f"{len(missing)} of {len(TARGET_SPECIES)} species are not in this BirdNET model.")
        print("Near matches in the label list, to help pick an alias:\n")
        for spec in missing:
            genus = spec.scientific_name.split()[0].lower()
            words = [w for w in spec.common_name.lower().replace("-", " ").split() if len(w) > 3]
            hits = [
                label
                for label in service.labels
                if label.lower().startswith(genus)
                or any(w in label.lower() for w in words)
            ]
            print(f"  {spec.common_name}:")
            for label in hits[:10] or ["    (nothing similar found)"]:
                print(f"    {label}")
            print()
        return 1

    print(f"All {len(TARGET_SPECIES)} target species are available in this model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
