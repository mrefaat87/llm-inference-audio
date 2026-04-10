#!/bin/bash
# Deploy audio-reader to GitHub Pages
#
# Setup (one-time):
#   1. Create a new GitHub repo (e.g., llm-inference-audio)
#   2. Run: git remote add pages git@github.com:YOUR_USERNAME/llm-inference-audio.git
#      (from this audio-reader/ directory)
#
# Then run this script to deploy:
#   ./deploy.sh
#
# Your site will be available at: https://YOUR_USERNAME.github.io/llm-inference-audio/

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if git is initialized in this directory
if [ ! -d .git ]; then
    echo "Initializing git repo for audio-reader..."
    git init
    git checkout -b main
fi

# Check total size of MP3s
NARRATION_SIZE=$(du -sh narration/ 2>/dev/null | cut -f1 || echo "0")
PODCAST_SIZE=$(du -sh podcast/ 2>/dev/null | cut -f1 || echo "0")
echo "Narration size: $NARRATION_SIZE"
echo "Podcast size: $PODCAST_SIZE"

# Stage all files
git add index.html generate_podcast.py deploy.sh
[ -d narration ] && git add narration/
[ -d podcast ] && git add podcast/

echo ""
echo "Files staged:"
git status --short

echo ""
read -p "Commit and push? (y/N) " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

git commit -m "Update audio reader"

# Push to remote
if git remote | grep -q pages; then
    git push pages main
    echo ""
    echo "Deployed! Check your GitHub repo settings to enable Pages on the main branch."
else
    echo ""
    echo "No 'pages' remote found. Add one with:"
    echo "  git remote add pages git@github.com:YOUR_USERNAME/llm-inference-audio.git"
    echo "  git push pages main"
    echo ""
    echo "Then enable GitHub Pages in repo Settings > Pages > Source: main branch"
fi
