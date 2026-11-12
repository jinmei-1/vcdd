"""Boltz API credentials and client construction.

Rules this module exists to enforce:

* The API key is read from an **environment variable only**. It is never a
  config field, so it cannot be committed, cannot end up in
  ``config.resolved.yaml``, and cannot be echoed by ``vcdd config show``.
* Nothing here ever returns, logs or formats the key itself. Callers get
  :class:`BoltzCredentials`, which carries *whether* a key was found and which
  mode it is in, and hands the key to the SDK without exposing it.
* Test-mode keys are detected and surfaced. A test key runs the full request
  lifecycle and returns synthetic results with no GPU time and no billing, so a
  run that quietly used one must be labelled - a stage-2 table of test-mode
  numbers looks exactly like a real one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class BoltzUnavailable(RuntimeError):
    """The Boltz SDK is missing, or no API key is configured."""


def boltz_available() -> bool:
    """True when the ``boltz-api`` SDK can be imported."""
    try:
        import boltz_api  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def sdk_version() -> str | None:
    try:
        import boltz_api

        return getattr(boltz_api, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class BoltzCredentials:
    """Non-secret facts about the configured key. Safe to log and to persist."""

    env_var: str
    present: bool
    #: False when the key is a test-mode key: full lifecycle, synthetic results,
    #: no GPU, no billing.
    livemode: bool
    #: Last four characters only, so two keys can be told apart in a log without
    #: the log becoming a credential.
    suffix: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "api_key_env": self.env_var,
            "api_key_present": self.present,
            "livemode": self.livemode,
            "api_key_suffix": self.suffix,
        }

    def require(self) -> None:
        if not self.present:
            raise BoltzUnavailable(
                f"no Boltz API key found in ${self.env_var}. Export it first:\n"
                f"    export {self.env_var}='sk_bc_...'\n"
                f"Keys are issued at https://api.boltz.bio. A test-mode key "
                f"(sk_bc_..._test_...) runs the whole pipeline with synthetic results at no "
                f"cost, which is the right way to rehearse a campaign."
            )


def _is_test_key(key: str) -> bool:
    """Boltz test keys carry a ``test`` segment in the prefix (``sk_bc_..._test_...``)."""
    head = key[:32].lower()
    return "_test_" in head or head.startswith("sk_test")


def resolve_credentials(env_var: str = "BOLTZ_API_KEY") -> BoltzCredentials:
    """Inspect the environment without returning the key."""
    key = os.environ.get(env_var, "").strip()
    if not key:
        return BoltzCredentials(env_var=env_var, present=False, livemode=False)
    return BoltzCredentials(
        env_var=env_var,
        present=True,
        livemode=not _is_test_key(key),
        suffix=key[-4:],
    )


def make_client(
    env_var: str = "BOLTZ_API_KEY",
    request_timeout_s: float = 120.0,
    max_retries: int = 2,
    http_client: Any | None = None,
) -> Any:
    """Construct a ``boltz_api.Boltz`` client.

    ``http_client`` is the seam the tests drive: passing an ``httpx.Client`` with
    a ``MockTransport`` exercises the real SDK - its pagination, its retry
    policy, its response models - against a scripted API, which is the only way
    to test this code path without a key and without spending money.

    The SDK's own retry policy is left in place: it retries 429/408/409/5xx twice
    with exponential backoff and honours ``Retry-After``. Re-implementing that
    here would only get it wrong.
    """
    try:
        from boltz_api import Boltz
    except Exception as exc:  # noqa: BLE001
        raise BoltzUnavailable(
            "the Boltz SDK is not installed. Install it with:\n"
            "    pip install 'vcdd[boltz]'   # or: pip install 'boltz-api>=0.46'"
        ) from exc

    creds = resolve_credentials(env_var)
    kwargs: dict[str, Any] = {
        "timeout": request_timeout_s,
        "max_retries": max_retries,
    }
    if http_client is not None:
        kwargs["http_client"] = http_client
        # A mock transport needs *a* key to get past the SDK's own check; it is
        # never sent anywhere real.
        kwargs["api_key"] = os.environ.get(env_var) or "sk_bc_dummy_test_key"
    else:
        creds.require()
        kwargs["api_key"] = os.environ[env_var]
    return Boltz(**kwargs)
