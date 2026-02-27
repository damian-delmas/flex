#!/bin/bash
# publish.sh — push filtered dev branch to public GitHub + tag for PyPI release
# Private modules stripped, public gets a clean snapshot.
# PyPI publishing is handled by GitHub Actions on tag push.
#
# Usage:
#   ./publish.sh              # strip, push to public, push version tag
#   ./publish.sh --dry-run    # show what would be removed, don't push

set -euo pipefail

REMOTE="public"
BRANCH="_pub"
TARGET="main"

# Private paths to exclude from public
PRIVATE=(
    flex/modules/claude_chat
    flex/modules/docpac
    flex/modules/soma
    flex/compile/docpac.py
    flex/compile/markdown.py
    views/claude_chat
    views/docpac
    operations
    scripts
    tests/test_docpac.py
    tests/test_docpac_worker.py
    tests/test_fingerprint.py
    tests/test_markdown.py
    tests/test_soma_module.py
    tests/test_unified_sync.py
)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Verify remote exists
if ! git remote get-url "$REMOTE" &>/dev/null; then
    echo "Remote '$REMOTE' not found. Add it:"
    echo "  git remote add $REMOTE git@github.com:axpsystems/flex.git"
    exit 1
fi

# Verify clean working tree
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Working tree not clean. Commit or stash first."
    exit 1
fi

VERSION=$(grep '^version' pyproject.toml | head -1 | grep -oP '"\K[^"]+')

echo "Publishing dev → $REMOTE/$TARGET (v$VERSION)"
echo "Stripping ${#PRIVATE[@]} private paths"

if $DRY_RUN; then
    echo ""
    echo "Would remove:"
    for p in "${PRIVATE[@]}"; do
        if git ls-files "$p" | grep -q .; then
            echo "  $p"
        fi
    done
    echo ""
    echo "Dry run — no changes made."
    exit 0
fi

# Create orphan branch from dev
git checkout --orphan "$BRANCH" dev

# Remove private files from index (keep on disk via orphan reset)
for p in "${PRIVATE[@]}"; do
    git rm -r --cached "$p" 2>/dev/null || true
done

# Commit + tag
git commit -m "release v$VERSION"
git tag "v$VERSION"

# Push branch + tag
git push "$REMOTE" "$BRANCH:$TARGET" --force
git push "$REMOTE" "v$VERSION" --force

# Cleanup
git checkout -f dev
git branch -D "$BRANCH"
git tag -d "v$VERSION"

echo ""
echo "Published to $REMOTE/$TARGET — tag v$VERSION pushed, PyPI release triggered"
