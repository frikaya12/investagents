# Bloc D sur Kaggle : agents extracteurs de facteurs de risque

Grille figée : `experiments/blocD/factor_grid.json` (19 facteurs, empreinte SHA-256 vérifiée par
`tests/test_extract.py`). Jeu de données privé `frikaya/investagents-blocd`, notebook privé
`frikaya/investagents-extracteur-blocd` (2 GPU T4, Qwen2.5 7B via Ollama). La clé Kaggle est lue par
le client depuis `~/.kaggle/` ; elle n'est jamais copiée dans le dépôt.

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run python -m investagents.extract.build_dataset          # phrases candidates (≈ 2 h)
# pilote : 30 premiers projets (validation)
uv run python -m investagents.extract.make_kernel --mode pilot
cd kaggle/blocD && kaggle kernels push -p . --accelerator NvidiaTeslaT4
kaggle kernels output frikaya/investagents-extracteur-blocd -p experiments/blocD/kaggle_pilot
uv run python -m investagents.extract.analyze --run experiments/blocD/kaggle_pilot
# passage complet (validation → test → entraînement), nouvelle version du jeu de données :
cd data/processed/kaggle_blocD && kaggle datasets version -p . -m "jeu complet"
uv run python -m investagents.extract.make_kernel --mode full
cd kaggle/blocD && kaggle kernels push -p . --accelerator NvidiaTeslaT4
kaggle kernels output frikaya/investagents-extracteur-blocd -p experiments/blocD/kaggle_full1
```
