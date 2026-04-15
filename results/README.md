# Results Artifacts

This directory stores small report-ready artifacts generated from the How2Sign subset manifest.

Generated command:

```powershell
python scripts/generate_project_results.py --manifest data/processed/how2sign_subset.example.jsonl --output-dir results
```

When real How2Sign clips, CNN predictions, and Qwen outputs are available, regenerate these files from `data/processed/how2sign_subset.jsonl`.
