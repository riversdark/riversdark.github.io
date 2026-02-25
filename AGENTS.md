# Agent Guide for `riversdark.github.io`

This file orients AI agents working on the Hugo blog hosted here.

## Quick Facts
- Hugo + PaperMod site; content is generated via ox-hugo from Org files in the parent repo `~/git/notes`.
- Primary blog authoring is consolidated in `~/git/notes/write.org`; runbook/config notes live in `~/git/notes/20251121122049-ox_hugo.org`.
- Do **not** keep manual edits to generated posts/pages under `content/` or `public/`; re-export from Org sources instead.
- Org sources target this repo with `#+hugo_base_dir: src/riversdark.github.io`.
- Deploy via local build → `gh-pages` branch using `scripts/deploy-gh-pages.sh` (sets `public/` as a worktree for `gh-pages`, adds `.nojekyll`, commits, and pushes). Pages settings should point to `gh-pages` branch, root folder. No GitHub Actions deploy workflow remains.
- Theme is the PaperMod module pinned in `go.mod`; KaTeX is loaded via `layouts/partials/math.html` when `.Param "math"` is true.
- Math rendering is handled in-template (KaTeX auto-render delimiters), not via post-export content rewrite scripts.

## Layout & Content Sources
- `content/`: ox-hugo exports from the parent `~/git/notes` repo; regenerate by editing the Org files there and running `M-x org-hugo-export-wim-to-md` (or `org-hugo-export-to-md`). Set front matter via `#+hugo_*`/`#+EXPORT_*` in Org, not by editing the generated Markdown.
- `hugo.toml`: site metadata, menus, TOC/math defaults, taxonomies, and permalinks (`posts` use `/:sections/:year/:month/:slug/`).
- `layouts/partials/extend_head.html` → `math.html`: conditionally injects KaTeX assets from jsDelivr when math is enabled. Delimiters include `$$...$$`, `\\[...\\]`, `\\(...\\)` and environment pairs like `\\begin{align*}...\\end{align*}` / `\\begin{equation*}...\\end{equation*}`.
- `scripts/deploy-gh-pages.sh`: production publish entrypoint; builds with `hugo --gc --minify`, then commits/pushes `public/` worktree.
- `_vendor/`: optional vendored Hugo modules; recreate with `hugo mod vendor` if needed.
- `public/`: `gh-pages` worktree with deploy artifacts only.
- `public_test/`: local scratch output; not part of deploy workflow.

## Development Workflow
- For content changes: edit the Org sources in `~/git/notes` with `#+hugo_base_dir: src/riversdark.github.io`, export with ox-hugo, then build here with `hugo serve -D` for drafts or `hugo --gc --minify` for production parity.
- Deploy to GitHub Pages (no CI): run `./scripts/deploy-gh-pages.sh` from repo root. First run creates the `gh-pages` branch as a worktree at `public/`; subsequent runs rebuild, commit, and push only when there are changes. Overrides: `DEPLOY_BRANCH`, `DEPLOY_MSG`.
- Deploy/build does not run post-export math normalization; if math fails, fix Org math form or KaTeX delimiter configuration.
- Date precedence for ox-hugo posts: `DONE` logbook timestamp and `CLOSED` can override `EXPORT_DATE`; if publish date must come from Org front matter, keep `:EXPORT_DATE:` and avoid `CLOSED` for that post.
- For theme/config tweaks: adjust `hugo.toml` or add overrides under `layouts/`; update PaperMod with `hugo mod get -u` + `hugo mod tidy` (vendor if necessary). Deploy locally with the script (CI workflow removed).
- Assets: place new static assets under `static/` (or reference ones managed in the parent repo); keep generated artifacts and caches (`public/`, `_vendor/`, `.hugo_build.lock`) out of commits per `.gitignore`.

## Interaction Guidelines
- Respect the source-of-truth rule: never hand-edit generated Markdown or `public/` output; all narrative/content edits belong in the Org files upstream.
- If temporary edits are needed for debugging export/render issues, treat them as disposable and re-export from Org before finalizing.
- Keep `main` for source/config changes; publishing happens on `gh-pages` via the deploy script.
- Preserve existing URLs, menus, and permalinks unless intentionally changing navigation; confirm TOC/math flags via front matter when exporting.
- Keep changes small and ASCII-only unless a file already contains non-ASCII text.
