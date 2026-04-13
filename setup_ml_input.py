# creates ml_input folder: frustration + mental_demand / passive + active / low med high
# copies from All_Outputs into corresponding level folders (per task2_last_step 23-33)

import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_OUTPUTS = os.path.join(SCRIPT_DIR, "Data", "All_Outputs")
ML_INPUT = os.path.join(SCRIPT_DIR, "Data", "ml_input")

# P1 mental 4 med, frustration 6 med; P2 mental 2 low, frustration 2 low; P13 mental 4 med, frustration 6 med
PARTICIPANT_LABELS = {
    "P1": ("med", "med"),
    "P2": ("low", "low"),
    "P13": ("med", "med"),
}


def ensure_dirs():
    for demand in ("mental_demand", "frustration"):
        for cond in ("passive", "active"):
            for level in ("low", "med", "high"):
                d = os.path.join(ML_INPUT, demand, cond, level)
                os.makedirs(d, exist_ok=True)


def copy_data():
    for pid, (mental_level, frust_level) in PARTICIPANT_LABELS.items():
        task2 = os.path.join(ALL_OUTPUTS, pid, "Task2_merged")
        if not os.path.isdir(task2):
            print(f"Skip {pid}: no Task2_merged")
            continue
        for cond in ("passive", "active"):
            # match any case (Passive, passive, etc.)
            src = None
            for name in os.listdir(task2):
                if name.lower() == cond:
                    p = os.path.join(task2, name, "combined_same_row.csv")
                    if os.path.isfile(p):
                        src = p
                        break
            if not src:
                print(f"Skip {pid} {cond}: no combined_same_row.csv")
                continue
            fname = f"{pid}_{cond}.csv"
            dst_mental = os.path.join(ML_INPUT, "mental_demand", cond, mental_level, fname)
            dst_frust = os.path.join(ML_INPUT, "frustration", cond, frust_level, fname)
            shutil.copy2(src, dst_mental)
            shutil.copy2(src, dst_frust)
            print(f"Copied {pid} {cond} -> mental_demand/.../{mental_level}/, frustration/.../{frust_level}/")


def main():
    ensure_dirs()
    copy_data()
    print("Done. ML input at:", ML_INPUT)


if __name__ == "__main__":
    main()
