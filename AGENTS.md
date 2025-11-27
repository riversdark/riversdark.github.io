# Agent Guide for `riversdark.github.io`

This file orients AI agents working on the Hugo blog hosted here.

## Quick Facts
- Hugo + PaperMod site; content is generated via ox-hugo from Org files in the parent repo `~/git/notes`.
- Do **not** write or edit posts/pages directly under `content/` or `public/`; re-export from the Org sources instead.
- Org sources target this repo with `#+hugo_base_dir: src/riversdark.github.io` (e.g., `~/git/notes/20251126093328-blog_about.org`, `~/git/notes/20251126103038-54.org`).
- Deploy via local build → `gh-pages` branch using `scripts/deploy-gh-pages.sh` (sets `public/` as a worktree for `gh-pages`, adds `.nojekyll`, commits, and pushes). Pages settings should point to `gh-pages` branch, root folder.
- Theme is the PaperMod module pinned in `go.mod`; KaTeX is loaded via `layouts/partials/math.html` when `.Param "math"` is true.

## Layout & Content Sources
- `content/`: ox-hugo exports from the parent `~/git/notes` repo; regenerate by editing the Org files there and running `M-x org-hugo-export-wim-to-md` (or `org-hugo-export-to-md`). Set front matter via `#+hugo_*`/`#+EXPORT_*` in Org, not by editing the generated Markdown.
- `hugo.toml`: site metadata, menus, TOC/math defaults, taxonomies, and permalinks (`posts` use `/:sections/:year/:month/:slug/`).
- `layouts/partials/extend_head.html` → `math.html`: conditionally injects KaTeX assets from jsDelivr when math is enabled.
- `_vendor/`: optional vendored Hugo modules; recreate with `hugo mod vendor` if needed. `public/` is a `gh-pages` worktree and should contain only build artifacts.

## Development Workflow
- For content changes: edit the Org sources in `~/git/notes` with `#+hugo_base_dir: src/riversdark.github.io`, export with ox-hugo, then build here with `hugo serve -D` for drafts or `hugo --gc --minify` for production parity.
- Deploy to GitHub Pages (no CI): run `./scripts/deploy-gh-pages.sh` from repo root. First run creates the `gh-pages` branch as a worktree at `public/`; subsequent runs rebuild, commit, and push only when there are changes. Overrides: `DEPLOY_BRANCH`, `DEPLOY_MSG`.
- For theme/config tweaks: adjust `hugo.toml` or add overrides under `layouts/`; update PaperMod with `hugo mod get -u` + `hugo mod tidy` (vendor if necessary).
- Assets: place new static assets under `static/` (or reference ones managed in the parent repo); keep generated artifacts and caches (`public/`, `_vendor/`, `.hugo_build.lock`) out of commits per `.gitignore`.

## Interaction Guidelines
- Respect the source-of-truth rule: never hand-edit generated Markdown or `public/` output; all narrative/content edits belong in the Org files upstream.
- Keep `main` for source/config changes; publishing happens on `gh-pages` via the deploy script.
- Preserve existing URLs, menus, and permalinks unless intentionally changing navigation; confirm TOC/math flags via front matter when exporting.
- Keep changes small and ASCII-only unless a file already contains non-ASCII text.
