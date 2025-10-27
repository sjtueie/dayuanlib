#!/bin/bash
# Release script - Creates a git tag and triggers PyPI publish

set -e

if [ -z "$1" ]; then
    echo "Usage: ./scripts/release.sh <version>"
    echo "Example: ./scripts/release.sh 0.3.2"
    exit 1
fi

VERSION=$1
TAG="v${VERSION}"

echo "📦 Preparing release ${TAG}..."

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "❌ Tag ${TAG} already exists!"
    exit 1
fi

# Check if working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Working directory has uncommitted changes:"
    git status --short
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create annotated tag
echo "🏷️  Creating tag ${TAG}..."
git tag -a "$TAG" -m "Release ${VERSION}"

echo "✅ Tag ${TAG} created successfully!"
echo ""
echo "Next steps:"
echo "  1. Push tag: git push origin ${TAG}"
echo "  2. GitHub Actions will automatically publish to PyPI"
echo "  3. Monitor: https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/actions"
