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

## Four-task experiments

The comparison includes nine preset methods: a constant baseline, linear and regularized models, a decision tree, random forest, Extra Trees, histogram gradient boosting, and nearest neighbors. Grid and TPE searches for the forest and boosting models bring the total to thirteen training procedures per task.

Three outer folds evaluate each procedure, and three inner folds select parameters. House and student records use random folds. Adult uses stratified groups defined by fingerprints of the original fields. NBA validation moves forward through complete seasons. The inner folds follow the same rules. Imputation, encoding, and scaling are fitted only on the current training fold. Grid and TPE each make twelve evaluations over the same twelve-point parameter domain; TPE can revisit a combination.

| Task | Best mean result in this comparison | Outer-fold mean |
| --- | --- | --- |
| House prices | Boosting with grid or TPE search | RMSE USD 32,338.23 |
| Adult income | Boosting with grid or TPE search | AUC 0.8913 |
| NBA points | Preset random forest | RMSE 5.698 points |
| Student mathematics | Ridge | RMSE 13.545 points |

The rankings describe these three folds. The paper also reports variation across folds. The earlier single-split experiment and the nested experiment use different samples and configurations, so their scores do not directly measure tuning gains. Training and validation curves help diagnose fitting gaps. SHAP describes contributions within a fitted model; ablation removes fields, repeats tuning, and retrains the model. Both need to be read with possible substitution between fields in mind.

The question-answering evaluation checks field retrieval on sixteen fixed questions, fifteen of which include English field names. MRR is 0.96875. The quality of plans generated through the DeepSeek API has not been evaluated.

## Reproduce and check results

Launch the app once to prepare the Python environment, then double-click `Run-Research.command`. It installs experiment dependencies, runs the four tasks, checks metrics, produces figures, and rebuilds the paper and slides. Batch experiments make no DeepSeek API calls. Two task processes run in parallel, with one thread per model. Document compilation requires XeLaTeX and BibTeX.

Results are saved in `artifacts/research`, including fold row IDs, search histories, fitted models, row-level predictions, SHAP values, and ablations. Completed folds are reused. Archive old results before rerunning after a change to the experiment code.

Plotting uses Nature Skills at `~/Downloads/SKILLS/nature-skills-main/skills/nature-figure` by default. Set `NATURE_FIGURE_SKILL` to use another location for that skill directory.

Run these commands from the project root to check saved results or rebuild the documents:

```bash
# Check the earlier single-split results
codes/.venv/bin/python codes/check_results.py

# Recompute extended experiment metrics and check splits and SHAP additivity
codes/.venv/bin/python codes/tools/verify_research.py

# Compile the paper and Beamer slides
bash docs/build_docs.sh

# Check the current research deliverables before modifying files
shasum -a 256 -c codes/verification/research/files.sha256
```

## Files

| Path | Contents |
| --- | --- |
| `data` | Four datasets, source manifest, and processed CSV files |
| `codes` | Web app, field retrieval, DeepSeek interface, models, and verification code |
| `artifacts/research` | Models and row-level results from nested validation |
| `figures`, `tables` | Figures and result tables from the earlier single-split checks |
| `models` | Earlier training plans, splits, predictions, and demonstration records |
| `docs` | Paper, slides, sources, instructions, and archived versions |

Dataset sources and their verification status are recorded in Appendices C and D of the paper and in `data/manifest.json`.
