# Where data and artifacts live

Everything that is **not source code** for employers / git should live under **`Data/`**.

| Path | Contents |
|------|-----------|
| `Data/final_files_ML/` | `labels.txt` + `Frustration/*.csv` for training `--feature-source final_ml`. |
| `Data/examples/empatica_minute/` | Sample `eda.csv`, `temperature.csv`, `pulse-rate.csv` for minute-merge demos. |
| `Data/outputs/ml_results/` | Default merged / biomarker training artifacts. |
| `Data/outputs/ml_results_final_ml/` | Default `final_ml` checkpoints (`versions/` for versioned runs). |
| `Data/outputs/ml_server/` | Large legacy joblib dumps (optional). |
| `Data/documents/` | PDFs and similar (including vendor docs) consolidated here—not at `Data/` root. |
| `Data/ml_input`, `Data/biomarker_input`, … | Existing structured / research inputs. |

`.gitignore` ignores **`/Data/`** so clones stay small; copy your `Data/` tree from backup or regenerate exports locally.

Use **`scripts/git-add-code.sh`** to stage only code folders + root Python entrypoints + `README.md` / `requirements.txt`.
