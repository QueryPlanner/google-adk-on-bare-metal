#!/bin/sh
set -eu
umask 077

if [ "$#" -ne 6 ]; then
  echo "ERROR: Invalid deployment bootstrap identity."
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
case "$PHYSICAL_HOME" in
  /*) ;;
  *)
    echo "ERROR: Deployment home is invalid."
    exit 1
    ;;
esac
case "$PHYSICAL_HOME" in
  ""|*[!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./+-]*)
    echo "ERROR: Deployment home is invalid."
    exit 1
    ;;
esac
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
WORKTREE_CLEANUP_ARMED=0
LEASE_CREATED=0
KEEP_RELEASE=0

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

registered_release() {
  git_safe -C "$PROJECT_DIR" worktree list --porcelain \
    | grep -Fqx -- "worktree $RELEASE_DIR"
}

owned_release() {
  {
    IFS= read -r OWNER_SHA
    IFS= read -r OWNER_PROJECT
    IFS= read -r OWNER_RELEASE
    IFS= read -r OWNER_ORIGIN
    IFS= read -r OWNER_IMAGE
    IFS= read -r OWNER_COMPOSE
    IFS= read -r OWNER_STATE
    if IFS= read -r OWNER_EXTRA; then
      return 1
    fi
  } < "$OWNER_FILE"
  [ "$OWNER_SHA" = "$DEPLOY_SHA" ] \
    && [ "$OWNER_PROJECT" = "$PROJECT_DIR" ] \
    && [ "$OWNER_RELEASE" = "$RELEASE_DIR" ] \
    && [ "$OWNER_ORIGIN" = "$EXPECTED_ORIGIN" ] \
    && [ "$OWNER_IMAGE" = "$IMAGE_NAME" ] \
    && [ "$OWNER_COMPOSE" = "$DEPLOY_PROJECT" ] \
    && [ "$OWNER_STATE" = "$STATE_DIR" ]
}

remove_owned_partial_release() {
  env -i \
    "HOME=$PHYSICAL_HOME" \
    "PATH=$PATH" \
    LANG=C \
    LC_ALL=C \
    python3 -I -S -B -c '
import os
import shutil
import stat
import sys
from pathlib import Path

project = Path(sys.argv[1])
release = Path(sys.argv[2])
expected_name = f"{project.name}.release-{sys.argv[3]}-{sys.argv[4]}"
if (
    not release.is_absolute()
    or release != release.resolve(strict=False)
    or release.parent != project.parent
    or release.name != expected_name
    or not shutil.rmtree.avoids_symlink_attacks
):
    raise SystemExit("ERROR: Partial release cleanup proof failed.")
parent_descriptor = os.open(
    release.parent,
    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
)
try:
    metadata = os.stat(
        release.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise SystemExit("ERROR: Partial release cleanup proof failed.")
    shutil.rmtree(release.name, dir_fd=parent_descriptor)
finally:
    os.close(parent_descriptor)
' "$PROJECT_DIR" "$RELEASE_DIR" "$DEPLOY_RUN_ID" "$DEPLOY_RUN_ATTEMPT"
}

cleanup_bootstrap_release() {
  CLEANUP_STATUS=0
  if [ "$WORKTREE_CLEANUP_ARMED" -eq 1 ]; then
    if registered_release; then
      RELEASE_HEAD="$(git_safe -C "$RELEASE_DIR" rev-parse --verify HEAD)"
      if owned_release \
        && [ "$RELEASE_HEAD" = "$DEPLOY_SHA" ] \
        && ! git_safe -C "$RELEASE_DIR" symbolic-ref -q HEAD >/dev/null
      then
        if ! git_safe -C "$PROJECT_DIR" worktree remove \
          --force "$RELEASE_DIR"
        then
          CLEANUP_STATUS=1
        fi
      else
        CLEANUP_STATUS=1
      fi
    elif [ -e "$RELEASE_DIR" ]; then
      if ! owned_release || ! remove_owned_partial_release; then
        CLEANUP_STATUS=1
      fi
    fi
  fi
  if [ "$LEASE_CREATED" -eq 1 ] \
    && ! registered_release \
    && [ ! -e "$RELEASE_DIR" ]
  then
    rm -f -- "$OWNER_FILE" "$LOCK_FILE"
    if ! rmdir "$LEASE_DIR"; then
      CLEANUP_STATUS=1
    fi
  fi
  return "$CLEANUP_STATUS"
}

bootstrap_exit() {
  PRIMARY_STATUS=$?
  trap - EXIT INT TERM
  if [ "$KEEP_RELEASE" -eq 0 ]; then
    if ! cleanup_bootstrap_release; then
      echo "ERROR: Incomplete release worktree could not be cleaned."
      if [ "$PRIMARY_STATUS" -eq 0 ]; then
        PRIMARY_STATUS=1
      fi
    fi
  fi
  exit "$PRIMARY_STATUS"
}
trap bootstrap_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ -e "$PROJECT_DIR" ] && [ ! -d "$PROJECT_DIR/.git" ]; then
  echo "ERROR: Project path exists without a Git checkout."
  exit 1
fi
if [ -d "$PROJECT_DIR/.git" ]; then
  set +e
  env -i \
    "HOME=$PHYSICAL_HOME" \
    "PATH=$PATH" \
    LANG=C \
    LC_ALL=C \
    GIT_NO_REPLACE_OBJECTS=1 \
    python3 -I -S -B -c '
import os
import stat
import sys

path = sys.argv[1]
try:
    metadata = os.lstat(path)
except FileNotFoundError:
    raise SystemExit(0)
if (
    not stat.S_ISREG(metadata.st_mode)
    or metadata.st_uid != os.geteuid()
    or metadata.st_nlink != 1
    or metadata.st_size <= 0
):
    raise SystemExit(65)
if stat.S_IMODE(metadata.st_mode) != 0o600:
    raise SystemExit(64)
' "$PROJECT_DIR/.env"
  ENVIRONMENT_STATUS=$?
  set -e
  if [ "$ENVIRONMENT_STATUS" -eq 64 ]; then
    echo "ERROR: Legacy .env permissions block deployment."
    echo "Run on the VM: chmod 600 \"$PROJECT_DIR/.env\""
    exit 1
  fi
  if [ "$ENVIRONMENT_STATUS" -ne 0 ]; then
    echo "ERROR: Legacy .env ownership or file type is unsafe."
    exit 1
  fi
fi

env -i \
  "HOME=$PHYSICAL_HOME" \
  "PATH=$PATH" \
  LANG=C \
  LC_ALL=C \
  GIT_NO_REPLACE_OBJECTS=1 \
  python3 -I -S -B -c '
import sys

if sys.version_info[:2] != (3, 13):
    raise SystemExit("ERROR: Python 3.13 is required for deployment.")
'
command -v flock >/dev/null
if [ ! -d "$PROJECT_DIR/.git" ]; then
  echo "Project directory not found. Cloning..."
  git_safe clone "$EXPECTED_ORIGIN" "$PROJECT_DIR"
fi

ACTUAL_ORIGIN="$(git_safe -C "$PROJECT_DIR" remote get-url origin)"
case "$ACTUAL_ORIGIN" in
  "$EXPECTED_ORIGIN"|"$EXPECTED_ORIGIN.git") ;;
  *)
    echo "ERROR: Existing checkout has an unexpected origin."
    exit 1
    ;;
esac
if ! git_safe -C "$PROJECT_DIR" diff \
  --no-ext-diff --no-textconv --quiet --
then
  echo "ERROR: Tracked worktree changes block deployment."
  exit 1
fi
if ! git_safe -C "$PROJECT_DIR" diff \
  --cached --no-ext-diff --no-textconv --quiet --
then
  echo "ERROR: Staged changes block deployment."
  exit 1
fi

git_safe -C "$PROJECT_DIR" fetch --no-tags origin "$DEPLOY_SHA"
if ! git_safe -C "$PROJECT_DIR" cat-file -e "${DEPLOY_SHA}^{commit}"; then
  echo "ERROR: Deployment commit is unavailable."
  exit 1
fi
for TARGET_PATH in \
  compose.yaml \
  compose.candidate.yaml \
  src/agent/__init__.py \
  src/agent/compose_env.py \
  src/agent/deployment_adoption.py \
  src/agent/deployment_promotion.py \
  src/agent/deployment_retention.py \
  src/agent/deployment_state.py
do
  TARGET_ENTRY="$(git_safe -C "$PROJECT_DIR" ls-tree "$DEPLOY_SHA" -- "$TARGET_PATH")"
  SAVED_IFS=$IFS
  IFS='	 '
  set -- $TARGET_ENTRY
  IFS=$SAVED_IFS
  if [ "$#" -ne 4 ] \
    || [ "$4" != "$TARGET_PATH" ] \
    || [ "$2" != blob ]
  then
    echo "ERROR: Deployment commit is missing required files."
    exit 1
  fi
  case "$1" in
    100644|100755) ;;
    *)
      echo "ERROR: Deployment commit has an unsafe required file type."
      exit 1
      ;;
  esac
done

if [ -e "$RELEASE_DIR" ] || [ -e "$LEASE_DIR" ]; then
  echo "ERROR: Release worktree or lease path already exists."
  exit 1
fi
mkdir -m 700 "$LEASE_DIR"
LEASE_CREATED=1
install -m 600 /dev/null "$LOCK_FILE"
exec 8> "$LOCK_FILE"
if ! flock -n 8; then
  echo "ERROR: Release bootstrap lease is already held."
  exit 1
fi
{
  printf '%s\n' "$DEPLOY_SHA"
  printf '%s\n' "$PROJECT_DIR"
  printf '%s\n' "$RELEASE_DIR"
  printf '%s\n' "$EXPECTED_ORIGIN"
  printf '%s\n' "$IMAGE_NAME"
  printf '%s\n' "$DEPLOY_PROJECT"
  printf '%s\n' "$STATE_DIR"
} > "$OWNER_FILE"
chmod 600 "$OWNER_FILE"
printf 'BOOTSTRAP_LEASE_READY:%s:%s\n' \
  "$DEPLOY_RUN_ID" "$DEPLOY_RUN_ATTEMPT"

WORKTREE_CLEANUP_ARMED=1
git_safe -C "$PROJECT_DIR" worktree add \
  --no-checkout --detach "$RELEASE_DIR" "$DEPLOY_SHA"
git_safe -C "$RELEASE_DIR" checkout --detach "$DEPLOY_SHA"
RELEASE_SHA="$(git_safe -C "$RELEASE_DIR" rev-parse --verify HEAD)"
if [ "$RELEASE_SHA" != "$DEPLOY_SHA" ] \
  || git_safe -C "$RELEASE_DIR" symbolic-ref -q HEAD >/dev/null
then
  echo "ERROR: Release worktree does not match deployment commit."
  exit 1
fi
if ! git_safe -C "$RELEASE_DIR" diff \
  --no-ext-diff --no-textconv --quiet --
then
  echo "ERROR: Release worktree has tracked changes."
  exit 1
fi
if ! git_safe -C "$RELEASE_DIR" diff \
  --cached --no-ext-diff --no-textconv --quiet --
then
  echo "ERROR: Release worktree has staged changes."
  exit 1
fi
RELEASE_STATUS="$(
  git_safe -C "$RELEASE_DIR" status \
    --porcelain=v1 \
    --untracked-files=all \
    --ignored=matching
)"
if [ -n "$RELEASE_STATUS" ]; then
  echo "ERROR: Release worktree is not exact-clean."
  exit 1
fi
for TARGET_PATH in \
  compose.yaml \
  compose.candidate.yaml \
  src/agent/__init__.py \
  src/agent/compose_env.py \
  src/agent/deployment_adoption.py \
  src/agent/deployment_promotion.py \
  src/agent/deployment_retention.py \
  src/agent/deployment_state.py
do
  if [ ! -f "$RELEASE_DIR/$TARGET_PATH" ] \
    || [ -L "$RELEASE_DIR/$TARGET_PATH" ]
  then
    echo "ERROR: Release worktree has an unsafe required file type."
    exit 1
  fi
done
KEEP_RELEASE=1
