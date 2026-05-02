# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A static, single-file web app (`index.html`) that presents an "LLM Inference Reference" book in three reading modes:

- **Read** — plain HTML.
- **Listen** — single-narrator MP3 per section with paragraph-level highlight sync (`narration/{sec_id}.mp3` + `.json` timing).
- **Podcast** — two-voice conversational MP3 per section (`podcast/{sec_id}.mp3`).

The whole app ships as one HTML file plus pre-rendered MP3s. There is no build step and no server — `index.html` is opened directly (or served via GitHub Pages by `deploy.sh`). Audio is generated offline by Python scripts and committed alongside the HTML.

## Common commands

```bash
# List every section ID and word count
python3 generate_podcast.py --list

# Generate Listen + Podcast MP3s for all sections (skips files that already exist)
python3 generate_podcast.py --all
python3 generate_podcast.py --all --force                 # regenerate

# Just one section
python3 generate_podcast.py --all --sections tokenization embeddings

# Real conversational podcast (Claude CLI writes the script, then edge-tts voices it)
python3 generate_real_podcast.py --all                    # edge-tts
python3 generate_real_podcast.py --all --force            # regen audio, keep cached transcripts
python3 generate_real_podcast.py --all --force-transcript # also re-run Claude

# Deploy to the `pages` git remote (GitHub Pages)
./deploy.sh
```

Dependencies (no requirements file): `pip install edge-tts beautifulsoup4`.

There are no tests, no linter, and no package manifest.

## Architecture

### Section-as-unit invariant

Everything keys off **section IDs**. In `index.html`, each topic lives in `<div id="sec-{id}">` and is paired with a `<button class="nav-item" data-sec="{id}">`. The Python generators discover sections by scanning for `id^="sec-"` and produce `{narration,podcast}/{id}.mp3`. The JS audio engine builds the audio src as `narration/{currentSec}.mp3` / `podcast/{currentSec}.mp3`. **Renaming a section ID means renaming the MP3, JSON, and transcript files together** — there is no mapping layer.

Quiz sections (IDs like `quiz-ch1`) intentionally have no audio; the generator's `SKIP_CLASSES` filters out `quiz` content and the section will be skipped if it has no extractable text.

### Text extraction → audio pipeline

Both Python scripts share an HTML-to-text extractor that walks the immediate children of each `sec-*` div and:

- Skips classes in `SKIP_CLASSES` (`prev-next`, `resource-bar`, `quiz`, `breadcrumb`, `kbd-hint`, `new-badge`) and `VIZ_CLASSES` (visualization helpers like `viz-flow`, `fmt-row`, `token-legend`).
- Renders `<table>` to "Row 1: header: cell. ..." style speech (capped at 8 rows).
- Returns one paragraph per surviving child element.

If you add a new presentational class to `index.html` that should not be spoken, add it to `SKIP_CLASSES` or `VIZ_CLASSES` in **both** generator scripts — they each maintain their own copies.

### Listen mode highlight sync

`generate_podcast.generate_narration` joins paragraphs with ` ... ` separators, streams `edge_tts.Communicate(...).stream()`, captures `SentenceBoundary` events, and maps each sentence back to its paragraph by string-matching its position in the joined text. Sentence events are then merged into one entry per contiguous paragraph and written to `narration/{id}.json`.

The browser side (`getSectionElements` in `index.html`) re-walks the same section's children with the **same skip rules and the same `text.length > 10` filter** to produce a parallel DOM array. The JSON's `paragraphIndex` indexes into that array. **If you change extraction rules in Python, mirror them in `getSectionElements` or highlights will drift.**

### Real-podcast pipeline (`generate_real_podcast.py`)

1. Extract section text (same extractor).
2. Build a per-section prompt using `CHAPTER_MAP` (section → chapter) and `CHAPTER_ANALOGIES` (chapter-specific seed analogies aimed at the AWS-EM audience described in `AUDIENCE`).
3. Shell out to `claude -p <prompt>` to produce an `Alex:` / `Jenny:` script. Scripts are cached in `podcast/transcripts/{id}.txt` — `--force` keeps them, `--force-transcript` discards them.
4. Parse turns and voice each turn with edge-tts (`en-US-GuyNeural` for Alex, `en-US-JennyNeural` for Jenny).
5. Concatenate MP3 frames directly to `podcast/{id}.mp3`.

Adding a new section to a chapter requires updating both `CHAPTER_MAP` (so the right chapter prompt is picked) and, ideally, `CHAPTER_ANALOGIES` (so the seed analogies cover the new concept).

### Deploy

`deploy.sh` is opinionated: it stages only `index.html`, `generate_podcast.py`, `deploy.sh`, `narration/`, `podcast/` and pushes to a remote literally named `pages`. It does **not** stage screenshots, `generate_real_podcast.py`, or the transcripts directory. If a new asset must ship to GitHub Pages, add it to the explicit `git add` list in `deploy.sh`.
