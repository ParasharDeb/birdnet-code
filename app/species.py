"""The target species and the logic that maps them onto BirdNET's label set.

BirdNET labels look like ``Pavo cristatus_Indian Peafowl`` — scientific name, an
underscore, then the eBird common name. Both halves drift between taxonomy
versions (Brown Fish-Owl has been in both *Bubo* and *Ketupa*; eBird writes
"Crested Serpent-Eagle" with a hyphen), so each target carries a list of
accepted spellings and we resolve against whatever labels the loaded model
actually ships with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def normalize(name: str) -> str:
    """Fold the spelling differences that don't change which bird is meant."""
    name = name.lower().replace("-", " ").replace("'", "").replace("`", "")
    return re.sub(r"\s+", " ", name).strip()


@dataclass(frozen=True)
class TargetSpecies:
    key: str
    common_name: str
    scientific_name: str
    emoji: str
    scientific_aliases: tuple[str, ...] = ()
    common_aliases: tuple[str, ...] = ()
    # Scientific names of *additional* model labels that also count as this
    # target. Unlike aliases (first match wins), every one of these that exists
    # in the model is mapped — used where one real-world bird spans labels.
    also_counts: tuple[str, ...] = ()

    @property
    def all_scientific(self) -> set[str]:
        return {normalize(n) for n in (self.scientific_name, *self.scientific_aliases)}

    @property
    def all_common(self) -> set[str]:
        return {normalize(n) for n in (self.common_name, *self.common_aliases)}


TARGET_SPECIES: tuple[TargetSpecies, ...] = (
    TargetSpecies(
        key="indian_peafowl",
        common_name="Indian Peafowl",
        scientific_name="Pavo cristatus",
        emoji="\N{PEACOCK}",
        common_aliases=("Common Peafowl", "Blue Peafowl", "Peafowl"),
    ),
    TargetSpecies(
        key="alexandrine_parakeet",
        common_name="Alexandrine Parakeet",
        scientific_name="Psittacula eupatria",
        emoji="\N{PARROT}",
        scientific_aliases=("Palaeornis eupatria",),
    ),
    TargetSpecies(
        key="greater_adjutant",
        common_name="Greater Adjutant",
        scientific_name="Leptoptilos dubius",
        emoji="\N{BIRD}",
        common_aliases=("Greater Adjutant Stork", "Adjutant Stork", "Greater Adjutant-Stork"),
    ),
    TargetSpecies(
        key="crested_serpent_eagle",
        common_name="Crested Serpent-Eagle",
        scientific_name="Spilornis cheela",
        emoji="\N{EAGLE}",
        common_aliases=("Crested Serpent Eagle",),
    ),
    TargetSpecies(
        key="brown_fish_owl",
        common_name="Brown Fish-Owl",
        scientific_name="Ketupa zeylonensis",
        emoji="\N{OWL}",
        scientific_aliases=("Bubo zeylonensis",),
        common_aliases=("Brown Fish Owl", "Brown Fishowl"),
    ),
    TargetSpecies(
        key="black_drongo",
        common_name="Black Drongo",
        scientific_name="Dicrurus macrocercus",
        emoji="\N{BIRD}",
    ),
    TargetSpecies(
        key="black_kite",
        common_name="Black Kite",
        scientific_name="Milvus migrans",
        emoji="\N{BIRD}",
    ),
    TargetSpecies(
        key="house_sparrow",
        common_name="House Sparrow",
        scientific_name="Passer domesticus",
        emoji="\N{BIRD}",
    ),
    TargetSpecies(
        key="white_winged_duck",
        common_name="White-winged Duck",
        scientific_name="Asarcornis scutulata",
        emoji="\N{DUCK}",
        scientific_aliases=("Cairina scutulata",),
        common_aliases=("White-winged Wood Duck", "Deohah"),
    ),
    # BirdNET has no "domestic goose" class. Domestic geese descend from the
    # Graylag (European/Indian breeds) and the Swan Goose (Chinese/African
    # breeds) and sound like them, so either label counts as a domestic goose.
    TargetSpecies(
        key="domestic_goose",
        common_name="Domestic Goose",
        scientific_name="Anser anser",
        emoji="\N{GOOSE}",
        scientific_aliases=("Anser anser domesticus",),
        common_aliases=("Graylag Goose", "Greylag Goose"),
        also_counts=("Anser cygnoides",),
    ),
)

BY_KEY: dict[str, TargetSpecies] = {s.key: s for s in TARGET_SPECIES}


@dataclass
class Resolution:
    """How one target species mapped onto the loaded model's labels."""

    species: TargetSpecies
    label: str | None = None           # exact BirdNET label, e.g. "Pavo cristatus_Indian Peafowl"
    matched_on: str | None = None      # "scientific" | "common" | None
    model_common_name: str | None = None
    model_scientific_name: str | None = None
    extra_labels: list[str] = field(default_factory=list)  # from ``also_counts``

    @property
    def resolved(self) -> bool:
        return self.label is not None

    @property
    def all_labels(self) -> list[str]:
        return ([self.label] if self.label else []) + self.extra_labels


def _split(label: str) -> tuple[str, str]:
    sci, _, common = label.partition("_")
    return sci.strip(), common.strip()


def resolve_targets(labels: list[str]) -> dict[str, Resolution]:
    """Match every target species against the model's label list.

    Scientific name wins over common name: it is the more stable identifier and
    is not shared between unrelated birds the way "Fish Owl" might be.
    """
    by_scientific: dict[str, str] = {}
    by_common: dict[str, str] = {}
    for label in labels:
        sci, common = _split(label)
        by_scientific.setdefault(normalize(sci), label)
        by_common.setdefault(normalize(common), label)

    resolutions: dict[str, Resolution] = {}
    for target in TARGET_SPECIES:
        res = Resolution(species=target)
        for candidate in target.all_scientific:
            if candidate in by_scientific:
                res.label, res.matched_on = by_scientific[candidate], "scientific"
                break
        else:
            for candidate in target.all_common:
                if candidate in by_common:
                    res.label, res.matched_on = by_common[candidate], "common"
                    break
        if res.label:
            res.model_scientific_name, res.model_common_name = _split(res.label)
        for sci in target.also_counts:
            extra = by_scientific.get(normalize(sci))
            if extra and extra != res.label:
                res.extra_labels.append(extra)
        resolutions[target.key] = res
    return resolutions
