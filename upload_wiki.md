# GitHub Wiki Upload Instructions

The wiki pages have been created in the `wiki/` directory. To upload them to your GitHub repository's wiki:

## Method 1: Using Git (Recommended)

1. Clone the wiki repository:
```bash
git clone https://github.com/neophrythe/nexus-ai.wiki.git
cd nexus-ai.wiki
```

2. Copy the wiki files:
```bash
# Windows PowerShell
Copy-Item -Path "C:\Users\neoph\Desktop\GAMEAI\wiki\*" -Destination "." -Recurse

# Or manually copy all .md files from wiki/ folder to the cloned wiki repo
```

3. Commit and push:
```bash
git add .
git commit -m "Add comprehensive wiki documentation"
git push
```

## Method 2: Manual Upload via GitHub Web

1. Go to https://github.com/neophrythe/nexus-ai/wiki
2. Click "Create the first page" or "New Page"
3. For each file in the `wiki/` directory:
   - Create a new page with the same name (without .md extension)
   - Copy and paste the content
   - Save the page

## Wiki Structure Created

The following wiki pages have been created:

1. **Home.md** - Main wiki homepage with navigation
2. **Installation.md** - Comprehensive installation guide
3. **Quick-Start-Guide.md** - 5-minute quick start tutorial
4. **Game-Plugins.md** - Complete guide to creating game plugins
5. **Agent-Development.md** - In-depth agent development guide

## Additional Wiki Pages to Create

You can create these additional pages by following the same pattern:

### CLI-Commands.md
- Full CLI reference
- All commands with examples
- Configuration options

### Computer-Vision.md
- Object detection guide
- OCR implementation
- Template matching

### Reinforcement-Learning.md
- RL algorithms explained
- Training strategies
- Hyperparameter tuning

### Troubleshooting.md
- Common issues and solutions
- Platform-specific problems
- Performance optimization

### API-Reference.md
- Complete Python API docs
- Code examples
- Class references

### Examples.md
- Complete game examples
- Different game genres
- Advanced techniques

## Wiki Features to Enable

In your GitHub repository settings:

1. Go to Settings → Features
2. Enable "Wikis"
3. Optionally enable "Restrict editing to collaborators only"

## Custom Sidebar

Create a `_Sidebar.md` file in the wiki with:

```markdown
## Navigation

**Getting Started**
- [[Home]]
- [[Installation]]
- [[Quick Start Guide]]

**Core Concepts**
- [[Game Plugins]]
- [[Agent Development]]

**Guides**
- [[CLI Commands]]
- [[Computer Vision]]
- [[Reinforcement Learning]]

**Reference**
- [[API Reference]]
- [[Configuration]]
- [[Troubleshooting]]

**Community**
- [[Examples]]
- [[Community Plugins]]
- [[Contributing]]
```

## Custom Footer

Create a `_Footer.md` file with:

```markdown
---
[GitHub](https://github.com/neophrythe/nexus-ai) | [Issues](https://github.com/neophrythe/nexus-ai/issues) | [Discord](https://discord.gg/nexus) | [Documentation](https://github.com/neophrythe/nexus-ai/wiki)
```

## Notes

- Wiki pages support GitHub Flavored Markdown
- Use `[[Page Name]]` for internal wiki links
- Images can be uploaded directly to wiki or linked from repo
- The wiki has its own git repository that can be cloned and edited locally
- Changes to wiki don't trigger notifications by default

## Quick Copy Commands

```bash
# Create all wiki pages at once (after cloning wiki repo)
cp /mnt/c/Users/neoph/Desktop/GAMEAI/wiki/*.md .
git add .
git commit -m "Add complete wiki documentation"
git push
```