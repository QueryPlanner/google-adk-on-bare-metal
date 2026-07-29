"""Test-only clean-process driver for the private Docker registry proof."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence

from agent.deployment_promotion import main as promotion_main

REPOSITORY = "QueryPlanner/google-adk-on-bare-metal"
EXPECTED_ORIGIN = "https://github.com/QueryPlanner/google-adk-on-bare-metal"
_LOCAL_IMAGE_REPOSITORY = re.compile(
    r"127\.0\.0\.1:(?P<port>[0-9]{1,5})/"
    r"adk-promotion-[a-z0-9][a-z0-9-]{0,23}-[0-9a-f]{16}/agent\Z"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drive promotion against one loopback-only test registry.",
    )
    parser.add_argument("--image-repository", required=True)
    parser.add_argument("--expected-origin", required=True, choices=(EXPECTED_ORIGIN,))
    parser.add_argument("--repository", required=True, choices=(REPOSITORY,))
    parser.add_argument("promotion_arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the isolated test seam and preserve production status handling."""
    arguments = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    match = _LOCAL_IMAGE_REPOSITORY.fullmatch(arguments.image_repository)
    if match is None or not 1 <= int(match.group("port")) <= 65535:
        raise SystemExit("image repository must use the isolated loopback registry")
    if (
        not arguments.promotion_arguments
        or arguments.promotion_arguments[0] != "promote"
    ):
        raise SystemExit("the runtime driver only supports promotion")
    return promotion_main(
        [
            "promote",
            "--expected-origin",
            arguments.expected_origin,
            "--repository",
            arguments.repository,
            *arguments.promotion_arguments[1:],
        ],
        _image_repository=arguments.image_repository,
    )


if __name__ == "__main__":
    raise SystemExit(main())
