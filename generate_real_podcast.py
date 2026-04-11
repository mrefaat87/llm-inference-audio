#!/usr/bin/env python3
"""
Generate real conversational podcast audio using Claude (via CLI) + edge-tts.

Pipeline:
  1. Extract section text from index.html
  2. Send to `claude -p` to generate a two-host conversation script
  3. Parse the script into speaker turns
  4. Voice each turn with edge-tts (different voice per speaker)
  5. Concatenate into final MP3

Usage:
    python3 generate_real_podcast.py --sections tokenization   # One section
    python3 generate_real_podcast.py --all                      # All 33 sections
    python3 generate_real_podcast.py --list                     # List sections

Requires: claude CLI (Claude Code), edge-tts, beautifulsoup4
"""

import asyncio
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import edge_tts
except ImportError:
    sys.exit("Install edge-tts: pip install edge-tts")

try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("Install beautifulsoup4: pip install beautifulsoup4")

BASE_DIR = Path(__file__).parent
SOURCE = BASE_DIR / "index.html"
PODCAST_DIR = BASE_DIR / "podcast"
TRANSCRIPT_DIR = BASE_DIR / "podcast" / "transcripts"

# Voices for the two hosts
VOICE_HOST = "en-US-GuyNeural"       # Alex — the explainer
VOICE_COHOST = "en-US-JennyNeural"   # Jenny — the curious questioner

SYSTEM_PROMPT = """You are a podcast script writer. Convert the following technical content into a natural, engaging conversation between two hosts:

- **Alex** (male): The main explainer. Deeply understands the material. Speaks clearly and uses analogies.
- **Jenny** (female): The curious co-host. Asks smart questions, pushes for clarity, reacts genuinely, connects ideas to practical implications.

Rules:
1. This must feel like a REAL conversation — not two people reading a textbook. Include reactions ("Oh wow", "That's clever", "Wait, so..."), interruptions, building on each other's points.
2. Cover ALL the key concepts from the source material. Don't skip important details.
3. Use analogies and real-world comparisons to make technical concepts accessible.
4. Jenny should ask the questions that a smart engineer new to this topic would ask.
5. Keep it concise — aim for a 3-5 minute conversation when spoken aloud (~500-800 words).
6. Output ONLY the dialogue, no stage directions or notes.

Format each line EXACTLY as:
Alex: <what Alex says>
Jenny: <what Jenny says>

Do not use any other format. Start the conversation directly."""


# ── Text extraction (reused from generate_podcast.py) ──

SKIP_CLASSES = {'prev-next', 'resource-bar', 'quiz', 'breadcrumb', 'kbd-hint', 'new-badge'}
VIZ_CLASSES = {'viz-flow', 'viz-timeline', 'viz-grid', 'viz-mem', 'fmt-row', 'fmt-bits',
               'fmt-legend', 'token-legend', 'token-row', 'legend-item'}


def extract_sections(html_path: Path) -> dict[str, list[str]]:
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
            if classes & SKIP_CLASSES or classes & VIZ_CLASSES:
                continue
            tag = el.name
            if tag == 'table':
                text = table_to_text(el)
            else:
                text = el.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 10:
                paragraphs.append(text)
        sections[sec_id] = paragraphs
    return sections


def table_to_text(table) -> str:
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    rows = [r for r in table.find_all('tr') if r.find('td')]
    if not rows:
        return ''
    lines = []
    for row in rows[:8]:
        cells = [td.get_text(strip=True) for td in row.find_all('td')]
        parts = [f"{headers[j]}: {c}" if j < len(headers) else c for j, c in enumerate(cells)]
        lines.append('; '.join(parts))
    return '\n'.join(lines)


# ── Conversation generation via Claude CLI ──

def generate_conversation(section_text: str, section_title: str) -> str:
    """Call claude -p to generate a podcast conversation script."""
    prompt = f"""Here is a technical article section titled "{section_title}":

---
{section_text}
---

{SYSTEM_PROMPT}"""

    result = subprocess.run(
        ['claude', '-p', prompt, '--output-format', 'text'],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0:
        print(f"    Claude CLI error: {result.stderr[:200]}")
        return ""

    return result.stdout.strip()


def parse_conversation(script: str) -> list[dict]:
    """Parse 'Alex: ...' / 'Jenny: ...' lines into structured turns."""
    turns = []
    current_speaker = None
    current_text = []

    for line in script.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Match speaker prefix
        match = re.match(r'^(Alex|Jenny)\s*:\s*(.*)', line, re.IGNORECASE)
        if match:
            # Save previous turn
            if current_speaker and current_text:
                turns.append({
                    'speaker': current_speaker,
                    'text': ' '.join(current_text)
                })
            current_speaker = match.group(1).capitalize()
            current_text = [match.group(2)] if match.group(2) else []
        elif current_speaker:
            # Continuation of current speaker's turn
            current_text.append(line)

    # Don't forget the last turn
    if current_speaker and current_text:
        turns.append({
            'speaker': current_speaker,
            'text': ' '.join(current_text)
        })

    return turns


# ── Audio generation ──

async def generate_audio(turns: list[dict], output_path: Path):
    """Generate MP3 from conversation turns using edge-tts."""
    temp_dir = output_path.parent / '.temp'
    temp_dir.mkdir(exist_ok=True)

    temp_files = []
    for i, turn in enumerate(turns):
        voice = VOICE_HOST if turn['speaker'] == 'Alex' else VOICE_COHOST
        temp_path = temp_dir / f'{output_path.stem}_{i:03d}.mp3'

        communicate = edge_tts.Communicate(turn['text'], voice, rate="+5%")
        await communicate.save(str(temp_path))
        temp_files.append(temp_path)

    # Concatenate all clips
    with open(output_path, 'wb') as out:
        for tf in temp_files:
            with open(tf, 'rb') as inp:
                out.write(inp.read())

    # Cleanup
    for tf in temp_files:
        tf.unlink()
    try:
        temp_dir.rmdir()
    except OSError:
        pass


# ── Main ──

async def process_section(sec_id: str, paragraphs: list[str], section_names: dict):
    """Full pipeline for one section: text → conversation → audio."""
    title = section_names.get(sec_id, sec_id)
    output_mp3 = PODCAST_DIR / f'{sec_id}.mp3'
    transcript_path = TRANSCRIPT_DIR / f'{sec_id}.txt'

    # Step 1: Check if transcript already exists (skip Claude call)
    if transcript_path.exists():
        print(f'  [{sec_id}] Using cached transcript')
        script = transcript_path.read_text()
    else:
        # Generate conversation via Claude
        section_text = '\n\n'.join(paragraphs)
        print(f'  [{sec_id}] Generating conversation via Claude CLI...')
        script = generate_conversation(section_text, title)
        if not script:
            print(f'  [{sec_id}] Failed to generate conversation')
            return False
        # Cache the transcript
        transcript_path.write_text(script)

    # Step 2: Parse turns
    turns = parse_conversation(script)
    if len(turns) < 4:
        print(f'  [{sec_id}] Warning: only {len(turns)} turns parsed (expected more)')
        if len(turns) == 0:
            return False

    alex_turns = sum(1 for t in turns if t['speaker'] == 'Alex')
    jenny_turns = sum(1 for t in turns if t['speaker'] == 'Jenny')
    total_words = sum(len(t['text'].split()) for t in turns)
    print(f'  [{sec_id}] {len(turns)} turns ({alex_turns} Alex, {jenny_turns} Jenny, ~{total_words} words)')

    # Step 3: Generate audio
    print(f'  [{sec_id}] Generating audio...')
    await generate_audio(turns, output_mp3)
    size_kb = output_mp3.stat().st_size / 1024
    print(f'  [{sec_id}] -> {output_mp3.name} ({size_kb:.0f} KB)')
    return True


def get_section_names(html_path: Path) -> dict:
    """Extract section display names from nav buttons."""
    with open(html_path, encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    names = {}
    for btn in soup.find_all('button', class_='nav-item'):
        sec = btn.get('data-sec', '')
        if sec:
            names[sec] = btn.get_text(strip=True)
    return names


async def main():
    parser = argparse.ArgumentParser(
        description='Generate real conversational podcast using Claude CLI + edge-tts')
    parser.add_argument('--all', action='store_true', help='Generate all sections')
    parser.add_argument('--sections', nargs='*', help='Specific section IDs')
    parser.add_argument('--list', action='store_true', help='List sections')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if MP3 exists (keeps cached transcripts)')
    parser.add_argument('--force-transcript', action='store_true',
                        help='Also regenerate transcripts (re-run Claude)')
    args = parser.parse_args()

    if not any([args.all, args.sections, args.list]):
        parser.print_help()
        return

    sections = extract_sections(SOURCE)
    section_names = get_section_names(SOURCE)

    if args.list:
        for sid, paras in sections.items():
            words = sum(len(p.split()) for p in paras)
            name = section_names.get(sid, sid)
            print(f'  {sid:20s} {name:30s} ~{words:,} words')
        return

    PODCAST_DIR.mkdir(exist_ok=True)
    TRANSCRIPT_DIR.mkdir(exist_ok=True)

    target_ids = args.sections if args.sections else list(sections.keys())
    generated = 0

    for sec_id in target_ids:
        if sec_id not in sections:
            print(f'  Skipping {sec_id} (not found)')
            continue

        output = PODCAST_DIR / f'{sec_id}.mp3'
        transcript = TRANSCRIPT_DIR / f'{sec_id}.txt'

        if args.force_transcript and transcript.exists():
            transcript.unlink()

        if output.exists() and not args.force and not args.force_transcript:
            print(f'  [{sec_id}] Skipping (already exists)')
            continue

        success = await process_section(sec_id, sections[sec_id], section_names)
        if success:
            generated += 1

    print(f'\nDone! {generated} podcast episodes generated in {PODCAST_DIR}/')
    print(f'Transcripts cached in {TRANSCRIPT_DIR}/')


if __name__ == '__main__':
    asyncio.run(main())
