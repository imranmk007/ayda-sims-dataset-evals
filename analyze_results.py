import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 40)


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_subheader(title):
    print(f"\n--- {title} ---")


# ──────────────────────────────────────────────────────────────────────
#  VLGuard Analysis
# ──────────────────────────────────────────────────────────────────────
print_header("VLGUARD ANALYSIS")

vg = pd.read_csv("results_vlguard.csv")
vg["refused"] = vg["refused"].astype(str).str.strip().str.lower() == "true"
vg["image_safe"] = vg["image_safe"].astype(str).str.strip().str.lower() == "true"

total = len(vg)
print(f"Total samples: {total}")

refused_count = vg["refused"].sum()
print(f"Overall refusal rate: {refused_count}/{total} ({refused_count/total*100:.1f}%)")

print_subheader("Refusal rate by instruction type")
by_type = vg.groupby("instruction_type")["refused"].agg(["sum", "count"])
by_type["rate"] = (by_type["sum"] / by_type["count"] * 100).round(1)
by_type.columns = ["refused", "total", "rate%"]
print(by_type)

print_subheader("Confusion matrix (instruction_type × refused)")
unsafe = vg[vg["instruction_type"] == "unsafe"]
safe = vg[vg["instruction_type"] == "safe"]

tp = unsafe["refused"].sum()
fn = (~unsafe["refused"]).sum()
fp = safe["refused"].sum()
tn = (~safe["refused"]).sum()

print(f"                    Refused    Complied")
print(f"  Unsafe instr:     {tp:>6}     {fn:>6}   (TP / FN)")
print(f"  Safe instr:       {fp:>6}     {tn:>6}   (FP / TN)")

if tp + fp > 0:
    precision = tp / (tp + fp)
    print(f"\nPrecision (of refusals): {precision*100:.1f}%")
if tp + fn > 0:
    recall = tp / (tp + fn)
    print(f"Recall (of unsafe caught): {recall*100:.1f}%")
if tp + fp > 0 and tp + fn > 0:
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"F1: {f1*100:.1f}%")

false_refusal_rate = fp / len(safe) * 100 if len(safe) > 0 else 0
false_compliance_rate = fn / len(unsafe) * 100 if len(unsafe) > 0 else 0
print(f"\nFalse refusal rate (safe but refused): {false_refusal_rate:.1f}%")
print(f"False compliance rate (unsafe but complied): {false_compliance_rate:.1f}%")


print_subheader("Refusal rate by harmful_category")
by_cat = vg.groupby("harmful_category")["refused"].agg(["sum", "count"])
by_cat["rate%"] = (by_cat["sum"] / by_cat["count"] * 100).round(1)
by_cat.columns = ["refused", "total", "rate%"]
print(by_cat.sort_values("rate%", ascending=False))


print_subheader("Refusal rate by harmful_subcategory")
by_sub = vg.groupby("harmful_subcategory")["refused"].agg(["sum", "count"])
by_sub["rate%"] = (by_sub["sum"] / by_sub["count"] * 100).round(1)
by_sub.columns = ["refused", "total", "rate%"]
print(by_sub.sort_values("rate%", ascending=False))


print_subheader("Refusal rate by image_safe × instruction_type")
cross = vg.groupby(["image_safe", "instruction_type"])["refused"].agg(["sum", "count"])
cross["rate%"] = (cross["sum"] / cross["count"] * 100).round(1)
cross.columns = ["refused", "total", "rate%"]
print(cross)


print_subheader("Most common refusal keywords")
kw_counts = vg[vg["refused"]]["refusal_keyword"].value_counts()
print(kw_counts)


# ──────────────────────────────────────────────────────────────────────
#  VLJailbreakBench Analysis
# ──────────────────────────────────────────────────────────────────────
print_header("VLJAILBREAKBENCH ANALYSIS")

vjb = pd.read_csv("results_vljailbreakbench.csv")
vjb["refused"] = vjb["refused"].astype(str).str.strip().str.lower() == "true"

total_jb = len(vjb)
print(f"Total samples: {total_jb}")
print(f"NOTE: Dataset has 3,600 total samples; eval may be partial.\n")


refused_jb = vjb["refused"].sum()
print(f"Overall refusal rate: {refused_jb}/{total_jb} ({refused_jb/total_jb*100:.1f}%)")


print_subheader("Refusal rate by split")
by_split = vjb.groupby("split")["refused"].agg(["sum", "count"])
by_split["rate%"] = (by_split["sum"] / by_split["count"] * 100).round(1)
by_split.columns = ["refused", "total", "rate%"]
print(by_split)


print_subheader("Refusal rate by harmful_label")
by_label = vjb.groupby("harmful_label")["refused"].agg(["sum", "count"])
by_label["rate%"] = (by_label["sum"] / by_label["count"] * 100).round(1)
by_label.columns = ["refused", "total", "rate%"]
print(by_label)


print_subheader("Most common refusal keywords")
kw_jb = vjb[vjb["refused"]]["refusal_keyword"].value_counts()
print(kw_jb)


print_subheader("Refusal rate by split × harmful_label")
cross_jb = vjb.groupby(["split", "harmful_label"])["refused"].agg(["sum", "count"])
cross_jb["rate%"] = (cross_jb["sum"] / cross_jb["count"] * 100).round(1)
cross_jb.columns = ["refused", "total", "rate%"]
print(cross_jb)

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}")
