"""Build the evaluation set: 25 resident questions + 25 document questions + 10 that must be refused.

    python scripts/build_eval_set.py         # writes data/eval/questions.jsonl

Gold labels survive re-chunking, so the same set scores every chunk configuration:
  * resident question  -> (resident's source_id, profile section)
  * document question  -> (source_id, an evidence regex the supporting chunk must contain)
Resident questions are generated from the de-identified gold profiles with a fixed seed, and
their expected keyword is read from the record itself. Document questions are hand-written;
each evidence regex is checked to match at least one real chunk.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcqa.dbsql import query

OUT = ROOT / "data" / "eval" / "questions.jsonl"

# --------------------------------------------------------------------------- documents (hand-written)
# (id, question, document_type, source_id, evidence regex, answer keyword regex)
DOCS = [
    ("reg_post_fall", "What does the regulation require after a resident falls?", "regulation",
     "oreg_246_22", r"post-fall assessment", r"post-fall assessment"),
    ("reg_skin_weekly", "How often must a resident's wound be reassessed under the regulation?", "regulation",
     "oreg_246_22", r"reassessed at least weekly", r"weekly"),
    ("reg_pain", "What must happen when a resident's pain is not relieved by initial interventions?", "regulation",
     "oreg_246_22", r"pain is not relieved", r"assess"),
    ("reg_bed_rails", "What does the regulation require when bed rails are used?", "regulation",
     "oreg_246_22", r"bed rails are used", r"assess"),
    ("reg_weight", "Which weight changes must be assessed for residents?", "regulation",
     "oreg_246_22", r"Weight changes|weight change of 5 per cent", r"5 per cent|per cent"),
    ("reg_ipac_lead", "What is the role of the infection prevention and control lead?", "regulation",
     "oreg_246_22", r"infection prevention and control lead", r"infection prevention"),
    ("reg_continence", "What must the continence care and bowel management program provide?", "regulation",
     "oreg_246_22", r"continence care and bowel management", r"continence"),
    ("act_privacy", "What right do residents have to privacy in treatment?", "legislation",
     "fltca_2021", r"privacy in treatment", r"privacy"),
    ("act_whistle", "Are staff protected from retaliation for reporting concerns to an inspector?", "legislation",
     "fltca_2021", r"retaliat", r"retaliat"),
    ("act_report_director", "What must be immediately reported to the Director?", "legislation",
     "fltca_2021", r"immediately report", r"abuse|neglect|harm"),
    ("act_restraint", "When can a resident be restrained by a physical device?", "legislation",
     "fltca_2021", r"restrain", r"restrain"),
    ("rnao_hip_protectors", "Does the falls guideline recommend hip protectors?", "guideline",
     "rnao_falls_2017", r"hip protector", r"hip protector"),
    ("rnao_exercise", "What exercise interventions does the guideline recommend to prevent falls?", "guideline",
     "rnao_falls_2017", r"exercise", r"exercise|balance|strength"),
    ("rnao_med_review", "Should medications be reviewed to reduce fall risk?", "guideline",
     "rnao_falls_2017", r"medication review|review of medication", r"medication"),
    ("rnao_vitamin_d", "What does the falls guideline say about vitamin D?", "guideline",
     "rnao_falls_2017", r"vitamin D", r"vitamin D"),
    ("rnao_cam", "Which tool can be used to screen older adults for delirium?", "guideline",
     "rnao_ddd_2016", r"Confusion Assessment Method", r"Confusion Assessment Method|CAM"),
    ("rnao_gds", "What scale is used to screen for depression in older adults?", "guideline",
     "rnao_ddd_2016", r"Geriatric Depression Scale", r"Geriatric Depression Scale|GDS"),
    ("rnao_reposition", "How often should people at risk of pressure injuries be repositioned?", "guideline",
     "rnao_pressure_2024", r"every 2.4 hours", r"2.4 hours|two to four hours"),
    ("rnao_braden", "Which risk assessment tool is mentioned for pressure injury risk?", "guideline",
     "rnao_pressure_2024", r"Braden", r"Braden"),
    ("ipac_moments", "What are the moments for hand hygiene in long-term care?", "ipac_guidance",
     "pho_ipac_ltch", r"four moments|4 moments", r"moment|before|after"),
    ("ipac_ppe_order", "In what order should personal protective equipment be put on?", "ipac_guidance",
     "pho_ipac_ltch", r"putting on|donning", r"gown|mask|glove"),
    ("label_warfarin_bleed", "What are the signs of bleeding to watch for with warfarin?", "drug_label",
     "label_warfarin", r"Signs and symptoms of bleeding", r"bleed"),
    ("label_metformin_lactic", "What is the boxed warning for metformin?", "drug_label",
     "label_metformin_er_500_mg", r"lactic acidosis", r"lactic acidosis"),
    ("label_simvastatin_myopathy", "What muscle side effects can simvastatin cause?", "drug_label",
     "label_simvastatin", r"myopathy", r"myopathy|rhabdomyolysis|muscle"),
    ("label_lisinopril_angioedema", "What serious swelling reaction can lisinopril cause?", "drug_label",
     "label_lisinopril", r"angioedema", r"angioedema"),
]
assert len(DOCS) == 25

# --------------------------------------------------------------------------- must refuse
NEGATIVE_PUBLIC = ["What is the capital of France?", "How do I bake sourdough bread?",
                   "What is the Wi-Fi password for the staff lounge?", "Who won the 2022 World Cup?",
                   "What time is bingo night this week?"]
NEGATIVE_RESIDENT = ["Does this resident have a pet dog?", "What is this resident's favourite food?",
                     "What is the phone number of this resident's daughter?", "What room is this resident in?",
                     "What is this resident's religion?"]


# --------------------------------------------------------------------------- residents (generated)
def section(text: str, name: str) -> str:
    m = re.search(rf"^## {re.escape(name)}[^\n]*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else ""


def items(body: str) -> list[str]:
    return [l[2:] for l in body.splitlines() if l.startswith("- ")]


def head_term(item: str) -> str:
    """'Malignant neoplasm of colon (disorder) (since 2024-08-03)' -> 'Malignant neoplasm of colon'."""
    return re.split(r" \(|;", item)[0].strip()


def drug_word(item: str) -> str:
    """'24 HR Metformin hydrochloride 500 MG ...' -> 'Metformin'; 'Alendronic acid 10 MG' -> 'Alendronic acid'."""
    words = re.findall(r"[A-Za-z][A-Za-z-]+", re.sub(r"^\d+ HR ", "", item))
    name = next(w for w in words if len(w) >= 4 and w.upper() != w)
    i = words.index(name) + 1
    return f"{name} acid" if i < len(words) and words[i] == "acid" else name


def resident_questions(profiles: dict[int, str], rng: random.Random) -> list[dict]:
    plan = [("fall", 5), ("meds", 5), ("conditions", 4), ("allergies", 4), ("injuries", 4), ("encounters", 3)]
    ids = sorted(profiles)
    rng.shuffle(ids)
    used, out = set(), []

    def make(rid, kind, q, sec, kw):
        out.append({"id": f"res_{kind}_{rid}", "kind": "resident", "question": q, "resident_id": rid,
                    "document_types": None, "expected": {"source_id": f"resident_{rid}", "section": sec},
                    "answer_regex": re.escape(kw)})

    for kind, n in plan:
        for rid in ids:
            if n == 0:
                break
            if rid in used:
                continue
            t = profiles[rid]
            if kind == "fall":
                m = re.search(r"Latest Morse Fall Scale assessment: (\w+ Risk)", t)
                if m:
                    make(rid, kind, "What was this resident's most recent Morse Fall Scale risk level?",
                         "Fall risk and falls", m.group(1))
                else:
                    continue
            elif kind == "meds":
                meds = items(section(t, "Active medications"))
                if not meds:
                    continue
                w = drug_word(meds[0])
                make(rid, kind, f"Is this resident currently taking {w.lower()}?", "Active medications", w)
            elif kind == "conditions":
                conds = [c for c in items(section(t, "Active conditions")) if "(disorder)" in c]
                if not conds:
                    continue
                c = head_term(conds[0])
                make(rid, kind, f"Does this resident have an active diagnosis of {c.lower()}?", "Active conditions", c)
            elif kind == "allergies":
                al = items(section(t, "Allergies"))
                if not al:
                    continue
                make(rid, kind, "What is this resident allergic to?", "Allergies", head_term(al[0]))
            elif kind == "injuries":
                inj = items(section(t, "Injuries"))
                if not inj:
                    continue
                make(rid, kind, "What injuries are recorded for this resident?", "Injuries",
                     head_term(inj[0].split(": ", 1)[1]))
            elif kind == "encounters":
                enc = items(section(t, "Recent encounters"))
                if not enc:
                    continue
                make(rid, kind, "When was this resident's most recent encounter, and what was it for?",
                     "Recent encounters", enc[0][:10])
            used.add(rid)
            n -= 1
    return out


def main() -> None:
    import pandas as pd
    rng = random.Random(42)
    _, rows = query("SELECT resident_id, profile_text FROM workspace.rcqa.gold_resident_profiles")
    profiles = {int(r): t for r, t in rows}
    res = resident_questions(profiles, rng)
    assert len(res) == 25, len(res)

    chunks = pd.read_parquet(ROOT / "data" / "exports" / "chunk_embeddings_w200_o40.parquet",
                             columns=["source_id", "text"])
    docs = []
    for qid, q, dtype, sid, evidence, kw in DOCS:
        n = chunks[(chunks.source_id == sid) & chunks.text.str.contains(evidence, case=False, regex=True)].shape[0]
        assert n > 0, f"{qid}: evidence /{evidence}/ not found in {sid}"
        docs.append({"id": qid, "kind": "document", "question": q, "resident_id": None, "document_types": [dtype],
                     "expected": {"source_id": sid, "evidence": evidence}, "answer_regex": kw})

    neg_ids = [r["resident_id"] for r in res[:len(NEGATIVE_RESIDENT)]]
    negs = [{"id": f"neg_public_{i}", "kind": "negative", "question": q, "resident_id": None, "document_types": None,
             "expected": None, "answer_regex": None} for i, q in enumerate(NEGATIVE_PUBLIC)]
    negs += [{"id": f"neg_resident_{i}", "kind": "negative", "question": q, "resident_id": rid, "document_types": None,
              "expected": None, "answer_regex": None} for i, (q, rid) in enumerate(zip(NEGATIVE_RESIDENT, neg_ids))]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for item in res + docs + negs:
            f.write(json.dumps(item) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(res)} resident, {len(docs)} document, {len(negs)} must-refuse")


if __name__ == "__main__":
    main()
