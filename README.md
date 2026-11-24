to generate the website from the survey:

```
uv run --locked --project ../feed-forward-survey python scripts/import-survey.py
hugo --minify --gc
```