# TableLab: Chinese table questions and prediction

English · [中文](README.zh-CN.md)

TableLab is a machine learning course project. It retrieves local field descriptions for a question in Chinese, then calls the DeepSeek API to produce a JSON analysis plan. After the user confirms the fields and conditions, local code reads the table and runs the calculation or trains a model.

The project covers house prices, Adult income classification, NBA player points, and student mathematics scores. It includes saved experiments for model comparison, grid and TPE hyperparameter search, fitting diagnostics, SHAP, and feature-group ablation.

## Run the app

On macOS, extract the entire folder and double-click `TableLab-DeepSeek.command`. The first launch needs an internet connection and Python 3.11, 3.12, or 3.13. The launcher creates `codes/.venv`, installs dependencies, and opens the web app.

Enter your own DeepSeek API key when the terminal prompts you. Input stays hidden. The launcher stores the key in your macOS user Keychain and reads it on later launches. You can also set `PRIVATE_API_KEY` in your local copy of the launcher; leave it empty in any copy you publish or share.

Ask questions in the browser and keep the terminal open. If the browser does not open automatically, wait for the ready message and visit <http://127.0.0.1:8765>. Press `Control-C` in the terminal to stop the app.

If double-clicking reports a permission error, type `bash` followed by a space in the terminal, drag `TableLab-DeepSeek.command` into the window, and press Return.

## Paper and slides

- [Course paper, in Chinese](docs/paper.pdf)
- [Beamer slides, in Chinese](docs/presentation.pdf)
- [Experiment summary, in Chinese](docs/research_summary.md)

LaTeX and BibTeX sources are in `docs/source`. The file `docs/demo_on_mac.txt` contains demonstration instructions. Open `docs/classroom_replay.html` to replay fixed examples without API calls; new questions require the web app.

## Experiments

The four original tasks compare 21 model entries: linear methods, tree ensembles, neighbors, SVM, XGBoost, LightGBM, CatBoost, MLP, residual and gated MLPs, Deep & Cross, a feature Transformer, ELM, and an RBF network. A separate NBA task compares 15 entries on the same three-season history windows, including RNN, GRU, BiGRU, LSTM, BiLSTM, causal TCN, and a sequence Transformer. These are batch experiments; the web app retains its existing interactive methods.

Each trainable entry has grid and TPE searches. Most domains have eight configurations; unpenalized linear models compare two intercept settings. Three inner folds select parameters and three outer folds evaluate the procedure. Adult uses stratified groups of full-predictor fingerprints. NBA validation advances through complete seasons. Preprocessing and regression target scaling are fitted only on training data. Each feature ablation repeats grid search; five tabular networks also have structural controls. The gradient-trained networks run for 40 epochs.

| Task | Better grid result in this comparison | Outer-fold mean ± SD |
| --- | --- | --- |
| House prices | RBFNet | RMSE 31,202.70 ± 4,473.78 USD |
| Adult income | CatBoost | AUC 0.8926 ± 0.0023 |
| NBA points | Random forest | RMSE 5.687 ± 0.393 points |
| Student mathematics | Linear SVM | RMSE 13.540 ± 0.387 points |
| NBA next-season points | BiGRU | RMSE 2.888 ± 0.098 points |

Adult also has a subsequent test extension for the 21 grid or fixed entries. The model selected from CV before testing is CatBoost: test AUC 0.8924, with a 95% cluster-bootstrap interval of [0.8871, 0.8977]. The original test had already been used for the earlier 13 procedures, so this is not a new blind test. Test scores did not change parameters or thresholds. The other tasks still lack new independent test samples. The NBA history task uses different inputs and records from the original NBA task; their errors do not directly measure an architecture gain.

SHAP explains a boosting reference fixed before the expanded results, on all outer-validation records. Ablation measures performance after retraining with less information or a disabled component. Neither field attribution nor feature removal identifies real-world causal effects. Retrieval is still checked on 16 fixed questions, 15 containing English field names; real DeepSeek plan quality has not been evaluated. [Results and methods, in Chinese](docs/research_summary.md).

## Reproduce and check results

Launch the app once to prepare the Python environment, stop the web service, then run `Run-Expanded.command`. It installs the pinned research dependencies, runs nested search and ablations, exports predictions in float64, checks the results, generates figures, and rebuilds the paper and Madrid slides. Batch experiments make no DeepSeek API calls. Each model uses one CPU thread; completed jobs are reused under the frozen protocol.

Expanded results are in `artifacts/expanded`. The earlier thirteen-procedure comparison is in `artifacts/research`, and its Adult test is in `artifacts/holdout/adult`; those records remain unchanged. The new Adult test extension is stored separately under `artifacts/expanded/holdout_extension`. Archive incompatible results before changing a frozen training implementation.

Figures use the designated Nature Skills directory, `~/Downloads/SKILLS/nature-skills-main/skills/nature-figure`. Set `NATURE_FIGURE_SKILL` to use another location. Document compilation needs XeLaTeX, BibTeX, and Font Awesome 5.

```bash
# Recompute expanded metrics and check every split and search
codes/.venv/bin/python codes/tools/verify_expanded.py
codes/.venv/bin/python codes/tools/verify_expanded_holdout.py
codes/.venv/bin/python codes/tools/verify_expanded_shap.py

# Rebuild the paper and slides
bash docs/build_docs.sh

# Check the delivered files before modifying them
shasum -a 256 -c codes/verification/research/files.sha256
```

## Files

| Path | Contents |
| --- | --- |
| `data` | Four datasets, source manifest, and processed CSV files |
| `codes` | Web app, field retrieval, DeepSeek interface, models, and verification code |
| `artifacts/expanded` | Expanded search, ablations, models, predictions, and test extension |
| `artifacts/research` | Models and row-level results from nested validation |
| `figures`, `tables` | Figures and result tables from the earlier single-split checks |
| `models` | Earlier training plans, splits, predictions, and demonstration records |
| `docs` | Paper, slides, sources, instructions, and archived versions |

Dataset sources and their verification status are recorded in Appendices C and D of the paper and in `data/manifest.json`.
