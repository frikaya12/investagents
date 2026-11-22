# Bloc C sur Kaggle : comité complet, GPU gratuit (T4 x2)

Notebook privé : https://www.kaggle.com/code/frikaya/investagents-comite-blocc
Jeu de données privé : https://www.kaggle.com/datasets/frikaya/investagents-blocc
Coût : 0 € (quota GPU gratuit de Kaggle, environ 30 h par semaine).

## Ce que fait le notebook
1. Installe Ollama et lance un serveur par GPU (ports 11434 et 11435), télécharge `qwen2.5:7b`.
2. Extrait le code du comité (`investagents_code.tar.gz`) et installe `langgraph`.
3. Lance `committee_run/runner.py` :
   - essai sur les 10 premiers projets, avec le temps par projet ;
   - test de reconnaissance sur les 50 premiers projets ;
   - puis tous les autres projets, dans l'ordre alterné validation / test.
   Chaque projet est écrit dans `results.jsonl` dès qu'il est fini. Le notebook s'arrête
   proprement avant la limite des 12 h.

Sur Kaggle, il n'y a ni identifiant réel, ni nom de projet, ni note IEG. La correspondance
clé → projet reste en local, dans `experiments/blocC/sample.csv`.

## Commandes (PowerShell ; le CLI est installé avec `uv tool install kaggle --with six`)
```
$env:PATH = "$HOME\.local\bin;$env:PATH"
kaggle kernels status frikaya/investagents-comite-blocc     # RUNNING / COMPLETE / ERROR
kaggle kernels logs frikaya/investagents-comite-blocc       # journal en cours
kaggle kernels output frikaya/investagents-comite-blocc -p experiments/blocC/kaggle_run1
```
La dernière commande récupère `results.jsonl`, `recognition.jsonl`, `progress.json` et les
journaux d'Ollama. Elle ne fonctionne qu'une fois l'exécution terminée.

## Reprendre après une session incomplète
Pour reprendre, on publie les anciens résultats dans un jeu de données, puis on relance le
notebook avec `--resume-from <chemin>/results.jsonl` : les projets déjà notés sont sautés.

## Régénérer
```
uv run python -m investagents.committee_run.build_dataset   # même graine → même échantillon
uv run python -m investagents.committee_run.make_kernel
cd data/processed/kaggle_blocC; kaggle datasets version -p . -m "maj"
cd kaggle/blocC; kaggle kernels push -p . --accelerator NvidiaTeslaT4
```
