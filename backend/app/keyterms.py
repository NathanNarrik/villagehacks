"""Initial keyterm list (50 medical terms).

OWNED BY PERSON A. See HANDOFF_PERSON_A.md for the full spec.

The learning loop in `app/learning_loop.py` calls `load_initial_keyterms()` whenever
its sorted set is empty (i.e. fresh process). After the first successful pipeline
run, learned terms take over and this function is no longer queried.
"""
from __future__ import annotations


def load_initial_keyterms() -> list[str]:
    """Return ~50 common medications + medical terms.

    Suggested categories (cover the benchmark scripts in HANDOFF_PERSON_A.md):
        - Cardiovascular: lisinopril, atorvastatin, amlodipine, metoprolol, losartan, ...
        - Diabetes: metformin, insulin, glipizide, ...
        - Mental health: sertraline, escitalopram, bupropion, ...
        - Pain: ibuprofen, acetaminophen, tramadol, gabapentin, ...
        - Antibiotics: amoxicillin, azithromycin, doxycycline, ciprofloxacin, ...
        - GI: omeprazole, pantoprazole, ranitidine, ...

    Pulled into Scribe v2's `keywords` parameter on first call.
    """
    return [
        # Cardiovascular
        "lisinopril",
        "atorvastatin",
        "amlodipine",
        "metoprolol",
        "losartan",
        "hydrochlorothiazide",
        "clopidogrel",
        "warfarin",
        "apixaban",
        "aspirin",
        # Diabetes / endocrine
        "metformin",
        "insulin",
        "glipizide",
        "sitagliptin",
        "semaglutide",
        "empagliflozin",
        "glyburide",
        "levothyroxine",
        # Mental health / neuro
        "sertraline",
        "escitalopram",
        "bupropion",
        "fluoxetine",
        "duloxetine",
        "venlafaxine",
        "gabapentin",
        "pregabalin",
        # Pain / inflammation
        "ibuprofen",
        "acetaminophen",
        "tramadol",
        "naproxen",
        "meloxicam",
        "diclofenac",
        "celecoxib",
        # Antibiotics / anti-infectives
        "amoxicillin",
        "azithromycin",
        "doxycycline",
        "ciprofloxacin",
        "cephalexin",
        "clindamycin",
        "metronidazole",
        # GI / reflux
        "omeprazole",
        "pantoprazole",
        "ranitidine",
        "famotidine",
        "ondansetron",
        # Respiratory / allergy
        "albuterol",
        "montelukast",
        "fluticasone",
        "cetirizine",
        "loratadine",
        # Common medical terms / routes / frequencies
        "milligrams",
        "micrograms",
        "twice daily",
        "once daily",
        "oral",
        "intravenous",
    ]
