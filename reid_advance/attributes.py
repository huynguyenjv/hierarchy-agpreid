"""AG-ReID.v2 attribute schema shared by the dataset, model, and trainer."""

from __future__ import annotations

from collections import OrderedDict


# The MAT file stores one-vs-all indicators using 1=False and 2=True.
# The last field in every group is the dataset's explicit ``unknown`` class.
ATTRIBUTE_GROUPS = OrderedDict(
    [
        ("gender", ("gendermale", "genderfemale", "genderunknown")),
        ("age", ("ageyoung", "agemiddle", "ageold", "ageunknown")),
        (
            "height",
            ("heightchild", "heightshort", "heightmedium", "heighttall", "heightunknown"),
        ),
        ("weight", ("weightthin", "weightmedium", "weightfat", "weightunknown")),
        (
            "ethnic",
            ("ethnicwhite", "ethnicblack", "ethnicasian", "ethnicindia", "ethnicunknown"),
        ),
        (
            "haircolor",
            (
                "haircolorblack",
                "haircolorbrown",
                "haircolorwhite",
                "haircolorred",
                "haircolorgray",
                "haircoloroccluded",
                "haircolorunknown",
            ),
        ),
        (
            "hairstyle",
            (
                "hairstylebald",
                "hairstyleshort",
                "hairstylemedium",
                "hairstylelong",
                "hairstylehorsetail",
                "hairstyleunknown",
            ),
        ),
        ("beard", ("beardon", "beardoff", "beardunknown")),
        ("moustache", ("moustacheon", "moustacheoff", "moustacheunknown")),
        ("glasses", ("glassesnormal", "glassessun", "glassesoff", "glassesunknown")),
        ("head", ("headhat", "headscarf", "headneckless", "headoccluded", "headunknown")),
        (
            "upper",
            (
                "uppertshirt",
                "upperblouse",
                "uppersweater",
                "uppercoat",
                "upperbikini",
                "uppernaked",
                "upperdress",
                "upperuniform",
                "uppershirt",
                "uppersuit",
                "upperhoodie",
                "uppercardiga",
                "upperunknown",
            ),
        ),
        (
            "lower",
            (
                "lowerjeans",
                "lowerleggins",
                "lowerpants",
                "lowershorts",
                "lowerskirt",
                "lowerbikini",
                "lowerdress",
                "loweruniform",
                "lowersuit",
                "lowerunknown",
            ),
        ),
        (
            "feet",
            (
                "feetsportshoe",
                "feetclassicshoe",
                "feethighheels",
                "feetboots",
                "feetsandals",
                "feetnothing",
                "feetunknown",
            ),
        ),
        (
            "bag",
            (
                "bagnormal",
                "bagbackpack",
                "baghandbag",
                "bagrolling",
                "bagumbrella",
                "bagsportif",
                "bagmarket",
                "bagnothing",
                "bagunknown",
            ),
        ),
    ]
)

ATTRIBUTE_NAMES = tuple(ATTRIBUTE_GROUPS)
ATTRIBUTE_CLASS_COUNTS = {name: len(fields) for name, fields in ATTRIBUTE_GROUPS.items()}
ATTRIBUTE_IGNORE_INDEX = -1
