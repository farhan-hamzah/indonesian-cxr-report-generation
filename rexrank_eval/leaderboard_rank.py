"""Rank one run's ReXrank results (study level) among the rexrank.ai leaderboard models.
usage (from rrg_indo/rexrank_eval):
    python leaderboard_rank.py ../outputs/rexrank_leaderboard/rexrank_tables_2026-10-16.json ../outputs/chexone_full/rexrank/chexone_full__rexrank.json
"""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
lb = json.load(open(sys.argv[1], encoding="utf-8"))
ours = json.load(open(sys.argv[2]))
ours = ours.get("results", ours)  # rexrank.py's <run>__rexrank.json or a bare results dict
cols = ["CRIMSON", "1/RadCliQ-v1", "BLEU", "BertScore", "SembScore", "RadGraph", "RaTEScore", "GREEN"]
ourkey = {"CRIMSON": "crimson", "1/RadCliQ-v1": "1/radcliq_v1", "BLEU": "bleu2", "BertScore": "bertscore", "SembScore": "semb_score",
          "RadGraph": "radgraph", "RaTEScore": "ratescore", "GREEN": "green"}
def f(x):
    try: return float(x)
    except: return None
for ds, split in [("MIMIC", "test_mimic"), ("IU", "test_openi"), ("CheXpert", "test_chexpert")]:
    for tb, track in [("test", "findings"), ("valid", "findings_impression")]:
        rows = lb[f"{tb}Results{ds}"]
        models = [(r[1].split(" ")[0] if False else r[1], dict(zip(cols, map(f, r[2:])))) for r in rows]
        m = ours[split][track]["study"]
        print(f"\n### {ds} {track} ({len(models)} models)")
        # _cxrm: BERTScore as CXR-Report-Metric's default config computes it, and RadCliQ-v1 with it
        for c, key in [(c, ourkey[c]) for c in cols] + [("BertScore", "bertscore_cxrm"), ("1/RadCliQ-v1", "1/radcliq_v1_cxrm")]:
            if key not in m: continue
            v = m[key]
            vals = sorted([s[c] for _, s in models if s[c] is not None], reverse=True)
            if not vals: continue
            rank = 1 + sum(x > v for x in vals)
            label = c + (" (cxrm)" if key.endswith("_cxrm") else "")
            print(f"  {label:<20} ours {v:.3f}  rank {rank}/{len(vals)+1}  (best {vals[0]:.3f}, median {vals[len(vals)//2]:.3f}, worst {vals[-1]:.3f})")
        top = sorted(models, key=lambda t: -(t[1]["1/RadCliQ-v1"] or 0))
        print("  by 1/RadCliQ:", "; ".join(f"{n} {s['1/RadCliQ-v1']}" for n, s in top))
