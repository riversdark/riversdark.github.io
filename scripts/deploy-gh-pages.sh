#!/usr/bin/env bash
set -euo pipefail

branch="${DEPLOY_BRANCH:-gh-pages}"
message="${DEPLOY_MSG:-Publish $(date -Iseconds)}"
repo_root="$(git rev-parse --show-toplevel)"
worktree_dir="${repo_root}/public"

cd "${repo_root}"

ensure_worktree() {
  git worktree prune

  if [ -d "${worktree_dir}/.git" ] && git -C "${worktree_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    current_branch="$(git -C "${worktree_dir}" symbolic-ref --short HEAD 2>/dev/null || true)"
    if [ "${current_branch}" != "${branch}" ]; then
      echo "public/ exists as a worktree but is on branch '${current_branch}' (expected '${branch}')." >&2
      exit 1
    fi
    return
  fi

  rm -rf "${worktree_dir}"

  if git show-ref --verify --quiet "refs/heads/${branch}"; then
    git worktree add -f "${worktree_dir}" "${branch}"
  else
    git worktree add --detach "${worktree_dir}"
    git -C "${worktree_dir}" switch --orphan "${branch}"
    git -C "${worktree_dir}" reset --hard
  fi
}

build_site() {
  hugo --gc --minify
  touch "${worktree_dir}/.nojekyll"
}

publish() {
  cd "${worktree_dir}"
  git add --all
  if git diff --cached --quiet; then
    echo "No changes to publish."
    return
  fi
  git commit -m "${message}"
  git push -u origin "${branch}"
}

ensure_worktree
build_site
publish
