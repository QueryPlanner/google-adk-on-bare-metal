#!/bin/sh
set -eu
umask 077

if [ "$#" -ne 6 ]; then
  echo "ERROR: Invalid release cleanup identity."
  exit 1
fi
DEPLOY_SHA=$1
IMAGE_DIGEST=$2
DEPLOY_RUN_ID=$3
DEPLOY_RUN_ATTEMPT=$4
PROJECT_NAME=$5
REPOSITORY=$6

case "$PROJECT_NAME" in
  ""|-*|*[!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-]*)
    echo "ERROR: Invalid deployment project name."
    exit 1
    ;;
esac
case "$REPOSITORY" in
  */*) ;;
  *)
    echo "ERROR: Invalid deployment repository."
    exit 1
    ;;
esac
case "$REPOSITORY" in
  ""|-*|*[!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./-]*)
    echo "ERROR: Invalid deployment repository."
    exit 1
    ;;
esac
if [ "${#DEPLOY_SHA}" -ne 40 ]; then
  echo "ERROR: Invalid deployment commit."
  exit 1
fi
case "$DEPLOY_SHA" in
  *[!0123456789abcdef]*)
    echo "ERROR: Invalid deployment commit."
    exit 1
    ;;
esac
case "$IMAGE_DIGEST" in
  sha256:*) ;;
  *)
    echo "ERROR: Invalid image digest."
    exit 1
    ;;
esac
DIGEST_HEX=${IMAGE_DIGEST#sha256:}
if [ "${#DIGEST_HEX}" -ne 64 ]; then
  echo "ERROR: Invalid image digest."
  exit 1
fi
case "$DIGEST_HEX" in
  *[!0123456789abcdef]*)
    echo "ERROR: Invalid image digest."
    exit 1
    ;;
esac
for RUN_VALUE in "$DEPLOY_RUN_ID" "$DEPLOY_RUN_ATTEMPT"; do
  case "$RUN_VALUE" in
    ""|*[!0123456789]*)
      echo "ERROR: Invalid deployment run identity."
      exit 1
      ;;
  esac
done

PHYSICAL_HOME="$(cd "$HOME" && pwd -P)"
PROJECT_DIR="$PHYSICAL_HOME/$PROJECT_NAME"
EXPECTED_ORIGIN="https://github.com/$REPOSITORY"
IMAGE_REPOSITORY="$(
  printf '%s' "ghcr.io/$REPOSITORY" | tr '[:upper:]' '[:lower:]'
)"
IMAGE_NAME="${IMAGE_REPOSITORY}@${IMAGE_DIGEST}"
PROJECT_SLUG="$(
  printf '%s' "$PROJECT_NAME" | tr '[:upper:].' '[:lower:]-'
)"
if [ "${#PROJECT_SLUG}" -gt 45 ]; then
  PROJECT_HASH="$(
    printf '%s' "$PROJECT_NAME" | sha256sum | cut -c1-8
  )"
  PROJECT_SLUG="$(printf '%.45s-%s' "$PROJECT_SLUG" "$PROJECT_HASH")"
fi
DEPLOY_PROJECT="adk-$PROJECT_SLUG"
STATE_DIR="$PHYSICAL_HOME/.local/state/$PROJECT_SLUG/deployment"
RELEASE_DIR="$PROJECT_DIR.release-$DEPLOY_RUN_ID-$DEPLOY_RUN_ATTEMPT"
LEASE_DIR="$RELEASE_DIR.lease"
OWNER_FILE="$LEASE_DIR/owner"
LOCK_FILE="$LEASE_DIR/lock"

git_safe() {
  env -i \
    "HOME=$PHYSICAL_HOME" \
    "PATH=$PATH" \
    LANG=C \
    LC_ALL=C \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_SYSTEM=/dev/null \
    GIT_TEMPLATE_DIR=/dev/null \
    GIT_NO_REPLACE_OBJECTS=1 \
    GIT_TERMINAL_PROMPT=0 \
    git \
      -c core.hooksPath=/dev/null \
      -c core.fsmonitor=false \
      "$@"
}

if [ ! -e "$LEASE_DIR" ] \
  && [ ! -L "$LEASE_DIR" ] \
  && [ ! -e "$RELEASE_DIR" ] \
  && [ ! -L "$RELEASE_DIR" ]
then
  exit 0
fi
if [ ! -d "$LEASE_DIR" ] || [ -L "$LEASE_DIR" ]; then
  echo "ERROR: Release lease is unavailable."
  exit 1
fi
env -i \
  "HOME=$PHYSICAL_HOME" \
  "PATH=$PATH" \
  LANG=C \
  LC_ALL=C \
  python3 -I -S -B -c '
import os
import stat
import sys

lease, owner, lock = sys.argv[1:]
for path, mode, directory in (
    (lease, 0o700, True),
    (owner, 0o600, False),
    (lock, 0o600, False),
):
    metadata = os.lstat(path)
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected_type(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != mode
        or (not directory and metadata.st_nlink != 1)
    ):
        raise SystemExit("ERROR: Release lease metadata is unsafe.")
' "$LEASE_DIR" "$OWNER_FILE" "$LOCK_FILE"
exec 9< "$LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: Release controller still holds the lease."
  exit 1
fi
{
  IFS= read -r OWNER_SHA
  IFS= read -r OWNER_PROJECT
  IFS= read -r OWNER_RELEASE
  IFS= read -r OWNER_ORIGIN
  IFS= read -r OWNER_IMAGE
  IFS= read -r OWNER_COMPOSE
  IFS= read -r OWNER_STATE
  if IFS= read -r OWNER_EXTRA; then
    exit 1
  fi
} < "$OWNER_FILE"
if [ "$OWNER_SHA" != "$DEPLOY_SHA" ] \
  || [ "$OWNER_PROJECT" != "$PROJECT_DIR" ] \
  || [ "$OWNER_RELEASE" != "$RELEASE_DIR" ] \
  || [ "$OWNER_ORIGIN" != "$EXPECTED_ORIGIN" ] \
  || [ "$OWNER_IMAGE" != "$IMAGE_NAME" ] \
  || [ "$OWNER_COMPOSE" != "$DEPLOY_PROJECT" ] \
  || [ "$OWNER_STATE" != "$STATE_DIR" ]
then
  echo "ERROR: Release ownership marker does not match."
  exit 1
fi
if ! git_safe -C "$PROJECT_DIR" worktree list --porcelain \
  | grep -Fqx -- "worktree $RELEASE_DIR"
then
  if [ ! -e "$RELEASE_DIR" ] && [ ! -L "$RELEASE_DIR" ]; then
    rm -f -- "$OWNER_FILE" "$LOCK_FILE"
    rmdir "$LEASE_DIR"
    exit 0
  fi
  echo "ERROR: Exact release worktree is not registered."
  exit 1
fi
if [ "$(git_safe -C "$RELEASE_DIR" rev-parse --verify HEAD)" != "$DEPLOY_SHA" ] \
  || git_safe -C "$RELEASE_DIR" symbolic-ref -q HEAD >/dev/null
then
  echo "ERROR: Exact detached release identity does not match."
  exit 1
fi
if ! git_safe -C "$RELEASE_DIR" diff \
  --no-ext-diff --no-textconv --quiet --
then
  echo "ERROR: Exact release worktree has tracked changes."
  exit 1
fi
if ! git_safe -C "$RELEASE_DIR" diff \
  --cached --no-ext-diff --no-textconv --quiet --
then
  echo "ERROR: Exact release worktree has staged changes."
  exit 1
fi
RELEASE_STATUS="$(
  git_safe -C "$RELEASE_DIR" status \
    --porcelain=v1 \
    --untracked-files=all \
    --ignored=matching
)"
if [ -n "$RELEASE_STATUS" ]; then
  echo "ERROR: Exact release worktree is not exact-clean."
  exit 1
fi
git_safe -C "$PROJECT_DIR" worktree remove "$RELEASE_DIR"
rm -f -- "$OWNER_FILE" "$LOCK_FILE"
rmdir "$LEASE_DIR"
