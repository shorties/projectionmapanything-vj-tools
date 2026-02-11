# Clean Installation Steps

The issue is a stale git URL in Scope's resolved dependencies.

## Option 1: Clear Plugin Cache (Recommended)

```bash
# Remove the corrupted resolved.txt
rm /root/.daydream-scope/plugins/resolved.txt

# Then install fresh
uv run daydream-scope install "https://github.com/shorties/projectionmapanything-vj-tools/archive/refs/heads/main.tar.gz"
```

## Option 2: Use --force

```bash
uv run daydream-scope install --force "https://github.com/shorties/projectionmapanything-vj-tools/archive/refs/heads/main.tar.gz"
```

## Option 3: Nuclear - Clear All Plugins

```bash
rm -rf /root/.daydream-scope/plugins/
mkdir -p /root/.daydream-scope/plugins/

# Reinstall from tarball
uv run daydream-scope install "https://github.com/shorties/projectionmapanything-vj-tools/archive/refs/heads/main.tar.gz"
```

## Verify Installation

```bash
# Check if plugin is listed
uv run daydream-scope list-plugins

# Or check entry points
uv run python -c "from importlib.metadata import entry_points; eps = entry_points(); print([e.name for e in eps.get('scope', [])])"
```
