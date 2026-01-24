# Validating Multi-Platform Docker Builds in Cloud Run

Multi-platform builds create a **manifest list** (index) and **platform-specific images** (amd64, arm64). Cloud Run deploys the manifest list but runs the platform-specific image, resulting in different digests in different places. This is expected.

## Verification Script

```bash
#!/usr/bin/env bash
set -euo pipefail

# Configuration - Set these 4 variables
PROJECT="your-gcp-project-id"
LOCATION="us-central1"
SERVICE_NAME="your-cloud-run-service-name"
VERSION_TAG="v0.4.1"

echo "=== Multi-Platform Build Validation ==="
echo ""

# Parse deployment details from Cloud Run
DEPLOYED_IMAGE=$(gcloud run services describe $SERVICE_NAME \
  --region $LOCATION \
  --format='value(spec.template.spec.containers[0].image)')

REGISTRY_URI=$(echo $DEPLOYED_IMAGE | cut -d'@' -f1)
IMAGE_NAME=$(echo $REGISTRY_URI | awk -F'/' '{print $NF}')
REPO=$(echo $REGISTRY_URI | awk -F'/' '{print $(NF-1)}')

REVISION_NAME=$(gcloud run services describe $SERVICE_NAME \
  --region $LOCATION \
  --format='value(status.latestReadyRevisionName)')

echo "Configuration:"
echo "  Service: $SERVICE_NAME"
echo "  Repository: $REPO"
echo "  Image: $IMAGE_NAME"
echo "  Revision: $REVISION_NAME"
echo ""

# Step 1: Get manifest list digest (deployed)
MANIFEST_DIGEST_URI=$(gcloud run services describe $SERVICE_NAME \
  --region $LOCATION \
  --format='value(spec.template.spec.containers[0].image)')
MANIFEST_DIGEST=$(echo $MANIFEST_DIGEST_URI | sed 's/.*@sha256://')
echo "Manifest list digest: sha256:$MANIFEST_DIGEST"

# Step 2: Get platform-specific digest (running)
PLATFORM_DIGEST_URI=$(gcloud run revisions describe $REVISION_NAME \
  --region $LOCATION \
  --format='value(spec.containers[0].image)')
PLATFORM_DIGEST=$(echo $PLATFORM_DIGEST_URI | sed 's/.*@sha256://')
echo "Platform digest:      sha256:$PLATFORM_DIGEST"
echo ""

# Step 3: Verify tag points to manifest list
TAG_DIGEST_URI=$(gcloud artifacts docker images describe \
  "${REGISTRY_URI}:${VERSION_TAG}" \
  --format="value(image_summary.digest)")
echo "Tag $VERSION_TAG → $TAG_DIGEST_URI"

if [[ "$TAG_DIGEST_URI" == "sha256:$MANIFEST_DIGEST" ]]; then
  echo "  ✓ Tag points to manifest list"
else
  echo "  ✗ Tag mismatch!"
  exit 1
fi

# Step 4: Verify manifest contains platform image
CONTAINED_DIGEST=$(docker manifest inspect \
  "${REGISTRY_URI}@sha256:${MANIFEST_DIGEST}" \
  | jq -r '.manifests[] | select(.platform.architecture=="amd64") | .digest')
echo "Manifest contains → $CONTAINED_DIGEST"

if [[ "$CONTAINED_DIGEST" == "sha256:$PLATFORM_DIGEST" ]]; then
  echo "  ✓ Manifest contains platform image"
else
  echo "  ✗ Platform mismatch!"
  exit 1
fi

echo ""
echo "=== Validation Passed ✓ ==="
echo ""
echo "Trace: Tag $VERSION_TAG → Manifest (${MANIFEST_DIGEST:0:12}...) → Platform (${PLATFORM_DIGEST:0:12}...)"
```

## Setup

**Docker authentication** (required for `docker manifest inspect`):
```bash
gcloud auth configure-docker ${LOCATION}-docker.pkg.dev
```

## Related Docs

- [CI/CD Workflow Guide](./cicd-setup.md)
