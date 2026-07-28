# ADR 0001: Private-TLS trace gateway

- Status: Accepted
- Date: 2026-07-28
- Scope: Optional single-VM trace export

## Context

Google ADK owns the process-wide OpenTelemetry provider. Even when structured
message and tool capture is disabled, raw exception events, status messages,
descriptions, identifiers, and custom attributes can enter the local telemetry
pipeline. Replacing ADK's provider or monkeypatching private span internals would
make this template brittle.

The supported boundary therefore needs to remove selected fields after they
leave the application process but before they leave the VM. It must not weaken
the application's rule that remote plaintext OTLP is rejected.

## Decision

Keep direct trace export as the default. Operators who need an outbound privacy
boundary can explicitly add `compose.trace-gateway.yaml`.

The gateway uses:

- one private `internal: true` Compose network shared only by the agent and
  Collector for OTLP traffic;
- HTTPS with a VM-owned CA and a server certificate whose SAN is
  `otel-collector`;
- bearer authentication loaded from one Compose secret file by both the agent
  and Collector;
- rejected Python OTLP credential-provider hooks plus direct Docker-DNS routing
  for the authenticated local hop;
- a dedicated host-side secret group added to both non-root containers, while
  mounting vendor credentials only into the Collector;
- a separate egress network joined only by the Collector;
- Collector-only Langfuse credentials, with no direct vendor fallback in the
  agent container; and
- a hardcoded HTTPS scheme and exact Langfuse trace path, with only the
  destination authority supplied by the operator; and
- a pinned Collector image and a fail-closed, deny-by-default trace pipeline.

Application readiness remains independent of both the Collector and Langfuse.
The Collector health endpoint is published only on host loopback and reports
process health, not delivery to the vendor.

## Rejected transport

A host-loopback Collector was rejected for the production topology:

- `127.0.0.1` inside the agent container is the container, not the VM host;
- Linux `host.docker.internal` and host-gateway routing do not make a host
  loopback-only listener a portable container destination;
- `network_mode: host` is Linux-specific and removes the existing network and
  port isolation; and
- `network_mode: service:...` couples the application and Collector lifecycle
  and changes the agent's published-port ownership.

Plaintext loopback remains useful for deterministic application tests, but it
is not the supported VM gateway transport.

## Outbound privacy contract

The pipeline drops spans containing links and drops every span event. It then:

- replaces all resource identity with fixed safe service values;
- clears instrumentation-scope attributes and normalizes scope name/version;
- canonicalizes known ADK operation families and maps unknown names to
  `agent.operation`;
- clears `status.message` and trace state;
- resets flags and all dropped-item counters;
- reconstructs only four currently verified integer ADK token-count attributes
  under fresh literal keys; and
- applies a second redaction allowlist; then
- reconstructs resource and scope envelopes from the allowlisted fields before
  batching and export.

Trace/span IDs, parentage, kind, timestamps, duration, status code, and the
allowed integer token counts remain. The reconstruction pass removes
`Resource.entity_refs`, `ResourceSpans.schema_url`, and
`ScopeSpans.schema_url`, plus the deprecated field-1000 scope envelope. These
fields are not exposed by the pinned transform context.

This policy is intentionally lossy. New ADK attributes are dropped until they
are explicitly reviewed. Do not enable Collector transform debug logging
because it can print the raw transform context.

“Fail-closed” here means unreviewed attribute and string-bearing fields are
removed for non-adversarial standard SDK instrumentation. It is not a DLP
boundary against compromised application code: retained IDs, timestamps,
integer counts, kind, and status code remain possible covert channels.

## Consequences

- Raw telemetry can still exist inside Python, ADK processors, the local OTLP
  request, and transient Collector receiver memory. This boundary governs what
  leaves the VM; it is not an in-process redaction guarantee.
- Application code that intentionally constructs hostile OTLP can encode data
  into retained structural or integer fields; the gateway does not defend
  against a compromised agent process.
- The Collector adds a second image, certificate and token rotation, roughly
  256 MiB of memory budget, and another upgrade cadence.
- The VM must provide Docker Compose 2.24.4 or newer plus one dedicated numeric
  group for group-readable bind-mounted secrets. This is more setup than
  running as root, but preserves non-root execution and per-container mounts.
- The in-memory sending queue is bounded and has no WAL. A crash or prolonged
  downstream outage can lose traces, but raw telemetry is not persisted on
  disk by this template.
- Collector or vendor failure is fail-open for the application. Monitor the
  Collector separately.
- Rollback is explicit: stop the overlay project, remove the gateway-only
  secrets and env files, and return to one documented direct-export mode.

## Verification

The pinned Collector binary must validate the tracked configuration. Pull
requests must also run a real isolated Docker test that sends crafted OTLP
protobuf through the gateway to a downstream TLS capture endpoint, proves
canaries do not leave the gateway, exercises receiver authentication and CA
rejection, and checks application health during Collector/downstream failure.

## Sources

- [Collector transform processor](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/v0.157.0/processor/transformprocessor)
- [Collector filter processor](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/v0.157.0/processor/filterprocessor)
- [Collector redaction processor](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/v0.157.0/processor/redactionprocessor)
- [Pinned groupbyattrs resource reconstruction](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/v0.157.0/processor/groupbyattrsprocessor/attribute_groups.go)
- [Collector security guidance](https://opentelemetry.io/docs/security/hosting-best-practices/)
- [OpenTelemetry exception conventions](https://opentelemetry.io/docs/specs/otel/trace/exceptions/)
