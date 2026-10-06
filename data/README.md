# Data files

| file | rows | used by |
|---|---:|---|
| `french_alpaca_20000.json` | 20,000 | detection prompts: `scripts/detection/run_cell.sh` (`PROMPTS`) |
| `french_alpaca_1000.json` | 1,000 | the evaluation prompts of the audit (`scripts/audit`) |
| `writingprompts_fr_500.json` | 500 | utility prompts: `scripts/utility/generate.py` |
| `wildchat_fr.json.gz` | 15,990 | fine-tuning corpus of the `ft` arm: `scripts/detection/finetune_attack.py` |

`french_alpaca_*.json` are JSON lists of instruction strings drawn from
[jpacifico/French-Alpaca-dataset-Instruct-55K](https://huggingface.co/datasets/jpacifico/French-Alpaca-dataset-Instruct-55K).
`writingprompts_fr_500.json` is a JSON list of 500 prompts of
[euclaise/writingprompts](https://huggingface.co/datasets/euclaise/writingprompts) translated to French, each prefixed with
"Réponds uniquement en français." and asking for a complete 250-300 word text.
`wildchat_fr.json.gz` is a gzip-compressed JSON list of `{"text": ...}` records: French conversations of
[allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat) rendered as plain text with the Llama 3.1
chat template, including that template's default system header (with the date the file was rendered);
the same file is used for every model.

The training corpora (LucieFr, AlpacaGPT4, OpenWebText) are streamed from the Hugging Face Hub; see
`scripts/data/build_local_corpora.py` for an offline copy.
