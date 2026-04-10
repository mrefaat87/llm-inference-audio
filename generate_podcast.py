#!/usr/bin/env python3
"""
Generate audio for the LLM Inference Reference using Microsoft edge-tts (free neural TTS).

Two output modes:
  narration/  — single narrator (Listen mode in the reader)
  podcast/    — two voices alternating paragraphs (Podcast mode in the reader)

Usage:
    pip install edge-tts beautifulsoup4

    python generate_podcast.py --all              # Generate both narration + podcast
    python generate_podcast.py --narration        # Single voice only
    python generate_podcast.py --podcast          # Two voices only
    python generate_podcast.py --list             # List available section IDs
    python generate_podcast.py --all --sections tokenization embeddings  # Specific sections only
"""

import asyncio
import argparse
import json
import re
from pathlib import Path

try:
    import edge_tts
except ImportError:
    print("Install edge-tts first: pip install edge-tts")
    raise

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Install beautifulsoup4 first: pip install beautifulsoup4")
    raise

BASE_DIR = Path(__file__).parent
SOURCE = BASE_DIR / "index.html"
NARRATION_DIR = BASE_DIR / "narration"
PODCAST_DIR = BASE_DIR / "podcast"

# High-quality Microsoft neural voices
VOICE_PRIMARY = "en-US-GuyNeural"      # Clear male narrator
VOICE_SECONDARY = "en-US-JennyNeural"  # Clear female narrator

# Classes to skip when extracting text
SKIP_CLASSES = {'prev-next', 'resource-bar', 'quiz', 'breadcrumb', 'kbd-hint', 'new-badge'}
VIZ_CLASSES = {'viz-flow', 'viz-timeline', 'viz-grid', 'viz-mem', 'fmt-row', 'fmt-bits',
               'fmt-legend', 'token-legend', 'token-row', 'legend-item'}


def extract_sections(html_path: Path) -> dict[str, list[str]]:
    """Extract text paragraphs per section from the HTML source."""
    with open(html_path, encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    sections = {}
    for div in soup.find_all('div', id=lambda x: x and x.startswith('sec-')):
        sec_id = div['id'].replace('sec-', '')
        paragraphs = []

        for el in div.children:
            if not hasattr(el, 'get'):
                continue
            classes = set(el.get('class', []))

            # Skip navigation, quizzes, visualizations
            if classes & SKIP_CLASSES:
                continue
            if classes & VIZ_CLASSES:
                continue

            tag = el.name
            text = ''

            if tag == 'h2':
                text = re.sub(r'\s*new\s*$', '', el.get_text(strip=True))
            elif 'subtitle' in classes:
                text = el.get_text(strip=True)
            elif 'sec-h' in classes:
                text = el.get_text(strip=True)
            elif tag == 'table':
                text = table_to_speech(el)
            elif 'deep-dive' in classes:
                raw = el.get_text(separator=' ', strip=True)
                text = 'Deep dive. ' + re.sub(r'^Deep Dive\s*[—-]?\s*', '', raw)
            elif tag in ('ul', 'ol'):
                items = [li.get_text(strip=True) for li in el.find_all('li')]
                text = '. '.join(items)
            else:
                text = el.get_text(separator=' ', strip=True)

            # Clean up technical syntax for natural speech
            text = re.sub(r'_', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            if len(text) > 10:
                paragraphs.append(text)

        sections[sec_id] = paragraphs

    return sections


def table_to_speech(table) -> str:
    """Convert an HTML table to natural spoken text."""
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    rows = table.find_all('tr')
    data_rows = [r for r in rows if r.find('td')]

    if not data_rows:
        return ''

    speech = 'Table. '
    for i, row in enumerate(data_rows[:8]):  # Cap at 8 rows for listening sanity
        cells = [td.get_text(strip=True) for td in row.find_all('td')]
        parts = []
        for j, cell in enumerate(cells):
            header = headers[j] + ': ' if j < len(headers) else ''
            parts.append(header + cell)
        speech += f'Row {i+1}: ' + '. '.join(parts) + '. '

    if len(data_rows) > 8:
        speech += f'({len(data_rows) - 8} more rows). '

    return speech


def map_sentence_to_paragraph(sentence_text: str, paragraphs: list[str],
                               full_text: str) -> int:
    """Find which paragraph a sentence belongs to by locating it in full_text."""
    # Find the sentence position in full_text
    pos = full_text.find(sentence_text)
    if pos == -1:
        return 0  # Fallback to first paragraph

    # Build paragraph start/end positions in full_text
    separator = ' ... '
    cursor = 0
    for i, para in enumerate(paragraphs):
        para_start = cursor
        para_end = cursor + len(para)
        if para_start <= pos < para_end:
            return i
        cursor = para_end + len(separator)

    return len(paragraphs) - 1  # Fallback to last


async def generate_narration(sec_id: str, paragraphs: list[str], output_path: Path):
    """Generate a single-narrator MP3 with timing data for highlight sync."""
    full_text = ' ... '.join(paragraphs)
    communicate = edge_tts.Communicate(full_text, VOICE_PRIMARY, rate="+5%")

    # Stream to capture both audio and sentence timing
    audio_chunks = []
    sentence_events = []

    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            audio_chunks.append(chunk['data'])
        elif chunk['type'] == 'SentenceBoundary':
            start = chunk['offset'] / 10_000_000  # 100ns ticks -> seconds
            duration = chunk['duration'] / 10_000_000
            para_idx = map_sentence_to_paragraph(
                chunk['text'], paragraphs, full_text)
            sentence_events.append({
                'start': round(start, 2),
                'end': round(start + duration, 2),
                'paragraphIndex': para_idx,
            })

    # Write MP3
    with open(output_path, 'wb') as f:
        for chunk in audio_chunks:
            f.write(chunk)

    # Consolidate sentence events into paragraph-level timing
    # Multiple sentences can belong to the same paragraph — merge them
    para_timing = []
    for evt in sentence_events:
        idx = evt['paragraphIndex']
        if para_timing and para_timing[-1]['paragraphIndex'] == idx:
            # Extend the existing paragraph entry
            para_timing[-1]['end'] = evt['end']
        else:
            para_timing.append({
                'start': evt['start'],
                'end': evt['end'],
                'paragraphIndex': idx,
            })

    # Write timing JSON alongside the MP3
    timing_path = output_path.with_suffix('.json')
    with open(timing_path, 'w') as f:
        json.dump(para_timing, f)


async def generate_podcast(sec_id: str, paragraphs: list[str], output_path: Path):
    """Generate a two-voice MP3 by alternating voices per paragraph.

    Two narrators trading off — more engaging than a single voice,
    like a co-hosted educational podcast.
    """
    temp_dir = output_path.parent / '.temp'
    temp_dir.mkdir(exist_ok=True)

    temp_files = []
    for i, para in enumerate(paragraphs):
        voice = VOICE_PRIMARY if i % 2 == 0 else VOICE_SECONDARY
        temp_path = temp_dir / f'{sec_id}_{i:03d}.mp3'
        communicate = edge_tts.Communicate(para, voice, rate="+5%")
        await communicate.save(str(temp_path))
        temp_files.append(temp_path)

    # Concatenate MP3 files (binary concat works for MP3 frames)
    with open(output_path, 'wb') as out:
        for tf in temp_files:
            with open(tf, 'rb') as inp:
                out.write(inp.read())

    # Cleanup temp files
    for tf in temp_files:
        tf.unlink()
    try:
        temp_dir.rmdir()
    except OSError:
        pass


async def generate_set(sections: dict, target_ids: list[str], output_dir: Path,
                       generate_fn, label: str):
    """Generate audio for a set of sections into the given directory."""
    output_dir.mkdir(exist_ok=True)
    generated = 0

    for sec_id in target_ids:
        if sec_id not in sections:
            print(f'  [{label}] Skipping {sec_id} (not found)')
            continue

        output = output_dir / f'{sec_id}.mp3'
        if output.exists():
            print(f'  [{label}] Skipping {sec_id} (already exists)')
            continue

        paragraphs = sections[sec_id]
        if not paragraphs:
            print(f'  [{label}] Skipping {sec_id} (no text)')
            continue

        print(f'  [{label}] Generating {sec_id} ({len(paragraphs)} paragraphs)...')
        await generate_fn(sec_id, paragraphs, output)
        size_kb = output.stat().st_size / 1024
        print(f'    -> {output.name} ({size_kb:.0f} KB)')
        generated += 1

    return generated


async def main():
    parser = argparse.ArgumentParser(
        description='Generate audio for LLM Inference Reference')
    parser.add_argument('--narration', action='store_true',
                        help='Generate single-voice narration (Listen mode)')
    parser.add_argument('--podcast', action='store_true',
                        help='Generate two-voice podcast (Podcast mode)')
    parser.add_argument('--all', action='store_true',
                        help='Generate both narration and podcast')
    parser.add_argument('--sections', nargs='*',
                        help='Only generate specific sections (by ID)')
    parser.add_argument('--list', action='store_true',
                        help='List available section IDs and exit')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if MP3 already exists')
    args = parser.parse_args()

    if not any([args.narration, args.podcast, args.all, args.list]):
        parser.print_help()
        print('\nExample: python generate_podcast.py --all')
        return

    sections = extract_sections(SOURCE)
    total_words = sum(
        sum(len(p.split()) for p in paras) for paras in sections.values()
    )

    if args.list:
        print(f'\n{len(sections)} sections, ~{total_words:,} words total:\n')
        for sid, paras in sections.items():
            words = sum(len(p.split()) for p in paras)
            print(f'  {sid:20s} {len(paras):3d} paragraphs, ~{words:,} words')
        return

    target_ids = args.sections if args.sections else list(sections.keys())

    # Delete existing files if --force
    if args.force:
        for sec_id in target_ids:
            for d in [NARRATION_DIR, PODCAST_DIR]:
                f = d / f'{sec_id}.mp3'
                if f.exists():
                    f.unlink()

    total = 0

    if args.narration or args.all:
        print(f'\n=== Generating narration (single voice: {VOICE_PRIMARY}) ===')
        n = await generate_set(sections, target_ids, NARRATION_DIR,
                               generate_narration, 'narration')
        total += n
        print(f'  {n} files generated in {NARRATION_DIR}/')

    if args.podcast or args.all:
        print(f'\n=== Generating podcast (two voices: {VOICE_PRIMARY} + {VOICE_SECONDARY}) ===')
        n = await generate_set(sections, target_ids, PODCAST_DIR,
                               generate_podcast, 'podcast')
        total += n
        print(f'  {n} files generated in {PODCAST_DIR}/')

    print(f'\nDone! {total} files generated total.')


if __name__ == '__main__':
    asyncio.run(main())
