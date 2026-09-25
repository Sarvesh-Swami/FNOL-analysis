"""Cognito authentication for the kinetic-api calls.

Everything here is driven by `.env` (see `.env.example`) - no credential is ever
written into the source.

The payload the browser sends to https://cognito-idp.<region>.amazonaws.com/ ...

    {"ChallengeName": "PASSWORD_VERIFIER",
     "ChallengeResponses": {"USERNAME", "PASSWORD_CLAIM_SECRET_BLOCK",
                            "TIMESTAMP", "PASSWORD_CLAIM_SIGNATURE"},
     "ClientId": "..."}

... is the *second* step of Cognito's SRP handshake. Those three claim values are
derived from the SRP exchange and are signed against that exact TIMESTAMP, so
they cannot be parked in `.env` and replayed - Cognito rejects them within
minutes ("Invalid signature for user" / NotAuthorizedException).

So the default flow here keeps the username and password in `.env` and performs
the whole handshake on every token refresh, producing exactly that payload
itself:

    InitiateAuth(USER_SRP_AUTH)  ->  challenge (SALT, SRP_B, SECRET_BLOCK)
    RespondToAuthChallenge(PASSWORD_VERIFIER)  ->  AuthenticationResult

AUTH_FLOW picks the strategy:

    auto              whatever the configured variables support (default)
    srp               USERNAME + PASSWORD, full SRP handshake  [recommended]
    password          USER_PASSWORD_AUTH (needs ALLOW_USER_PASSWORD_AUTH on the app client)
    refresh_token     REFRESH_TOKEN_AUTH from COGNITO_REFRESH_TOKEN
    password_verifier replays a captured challenge from .env    [expires in minutes]
    static            uses COGNITO_ACCESS_TOKEN verbatim        [expires in ~1 hour]
    none              disabled - requests go out unauthenticated, as before

Tokens are cached in memory and renewed a minute before ExpiresIn runs out,
preferring the cheap REFRESH_TOKEN_AUTH call over a fresh handshake.
"""

import os
import threading
import time

import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Renew this many seconds before the token actually expires, so an in-flight
# request never races the expiry.
REFRESH_MARGIN_SECONDS = 60

# Cognito's own wire protocol: a plain JSON POST to the regional endpoint with
# the operation named in an X-Amz-Target header. No AWS request signing is
# involved - these are unauthenticated public API calls.
_AMZ_CONTENT_TYPE = "application/x-amz-json-1.1"
_AMZ_TARGET_PREFIX = "AWSCognitoIdentityProviderService"


def _env(name, default=None):
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


def _env_bool(name, default=False):
    raw = _env(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in ("1", "true", "yes", "on")


class AuthError(RuntimeError):
    pass


class CognitoAuth:
    """Thread-safe access-token provider.

    Shared by the poller thread and the dashboard's request handler, so every
    token read goes through one lock: concurrent callers during a refresh wait
    for the single handshake instead of each starting their own.
    """

    def __init__(self):
        self.region = _env("COGNITO_REGION", "us-east-1")
        self.endpoint = _env(
            "COGNITO_IDP_ENDPOINT", f"https://cognito-idp.{self.region}.amazonaws.com/"
        )
        self.client_id = _env("COGNITO_CLIENT_ID")
        self.client_secret = _env("COGNITO_CLIENT_SECRET") or None
        self.user_pool_id = _env("COGNITO_USER_POOL_ID")
        self.username = _env("COGNITO_USERNAME")
        self.password = _env("COGNITO_PASSWORD")

        # Which of the two tokens the kinetic-api expects as the Bearer value.
        # Flip to "id" if the API rejects the access token.
        self.token_kind = (_env("AUTH_TOKEN_KIND", "access") or "access").lower()

        self.flow = (_env("AUTH_FLOW", "auto") or "auto").lower()

        self._lock = threading.Lock()
        self._token = None
        self._expires_at = 0.0
        self._refresh_token = _env("COGNITO_REFRESH_TOKEN") or None
        self._static_token = _env("COGNITO_ACCESS_TOKEN") or None
        self.last_error = None
        self.last_refresh_at = None
        self.refresh_count = 0

        if self.flow == "auto":
            self.flow = self._detect_flow()

        # Half-filled config is the likeliest mistake here. Without this, a
        # missing password just looks like the API going 401 on every tick.
        self.missing_config = self._missing_config()
        if self.missing_config:
            print(
                f"[auth] AUTH_FLOW={self.flow} is missing .env values: "
                f"{self.missing_config} - calls will go out unauthenticated until they are set."
            )

    def _detect_flow(self):
        if self.client_id and self.username and self.password and self.user_pool_id:
            return "srp"
        if self.client_id and self.username and self.password:
            return "password"
        if self.client_id and self._refresh_token:
            return "refresh_token"
        if self.client_id and _env("COGNITO_PASSWORD_CLAIM_SIGNATURE"):
            return "password_verifier"
        if self._static_token:
            return "static"
        return "none"

    def _missing_config(self):
        """Which .env values the chosen flow still needs. Empty list = good to go."""
        required = {
            "password": (
                ("COGNITO_CLIENT_ID", self.client_id),
                ("COGNITO_USERNAME", self.username),
                ("COGNITO_PASSWORD", self.password),
            ),
            "srp": (
                ("COGNITO_CLIENT_ID", self.client_id),
                ("COGNITO_USER_POOL_ID", self.user_pool_id),
                ("COGNITO_USERNAME", self.username),
                ("COGNITO_PASSWORD", self.password),
            ),
            "refresh_token": (
                ("COGNITO_CLIENT_ID", self.client_id),
                ("COGNITO_REFRESH_TOKEN", self._refresh_token),
            ),
            "password_verifier": (
                ("COGNITO_CLIENT_ID", self.client_id),
                ("COGNITO_USERNAME", self.username),
                ("COGNITO_PASSWORD_CLAIM_SECRET_BLOCK", _env("COGNITO_PASSWORD_CLAIM_SECRET_BLOCK")),
                ("COGNITO_PASSWORD_CLAIM_TIMESTAMP", _env("COGNITO_PASSWORD_CLAIM_TIMESTAMP")),
                ("COGNITO_PASSWORD_CLAIM_SIGNATURE", _env("COGNITO_PASSWORD_CLAIM_SIGNATURE")),
            ),
            "static": (("COGNITO_ACCESS_TOKEN", self._static_token),),
        }.get(self.flow, ())
        return [name for name, value in required if not value]

    @property
    def enabled(self):
        # A flow whose .env values are incomplete counts as off: it cannot
        # possibly succeed, and retrying it every 10s would just hammer Cognito
        # with a request that is guaranteed to fail.
        return self.flow != "none" and not self.missing_config

    # --- Cognito wire calls -------------------------------------------------

    def _cognito_post(self, target, payload):
        response = requests.post(
            self.endpoint,
            json=payload,
            headers={
                "Content-Type": _AMZ_CONTENT_TYPE,
                "X-Amz-Target": f"{_AMZ_TARGET_PREFIX}.{target}",
            },
            timeout=20,
        )
        if response.status_code != 200:
            # Cognito puts the useful part in the body, not the status line.
            raise AuthError(f"{target} failed (HTTP {response.status_code}): {response.text[:400]}")
        return response.json()

    def _secret_hash(self, username):
        """Only needed when the app client was created with a secret."""
        import base64
        import hashlib
        import hmac

        digest = hmac.new(
            self.client_secret.encode("utf-8"),
            (username + self.client_id).encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode()

    def _srp_login(self):
        """The full two-step SRP handshake, producing the PASSWORD_VERIFIER payload."""
        if not self.user_pool_id:
            raise AuthError("COGNITO_USER_POOL_ID is required for the srp flow")

        from pycognito.aws_srp import AWSSRP

        # pycognito is used purely for the SRP arithmetic (SRP_A, the HKDF-derived
        # password key and the claim signature). The HTTP calls stay here so the
        # requests match the documented Cognito endpoint one-for-one.
        srp = AWSSRP(
            username=self.username,
            password=self.password,
            pool_id=self.user_pool_id,
            client_id=self.client_id,
            pool_region=self.region,
            client_secret=self.client_secret,
        )

        auth_params = srp.get_auth_params()
        initiate = self._cognito_post(
            "InitiateAuth",
            {
                "AuthFlow": "USER_SRP_AUTH",
                "AuthParameters": auth_params,
                "ClientId": self.client_id,
            },
        )

        challenge_name = initiate.get("ChallengeName")
        if challenge_name != "PASSWORD_VERIFIER":
            raise AuthError(
                f"Expected a PASSWORD_VERIFIER challenge, got {challenge_name or 'no challenge'}"
            )

        challenge_responses = srp.process_challenge(
            initiate.get("ChallengeParameters", {}), auth_params
        )
        result = self._cognito_post(
            "RespondToAuthChallenge",
            {
                "ChallengeName": "PASSWORD_VERIFIER",
                "ChallengeResponses": challenge_responses,
                "ClientId": self.client_id,
            },
        )
        return self._unpack(result)

    def _password_login(self):
        auth_parameters = {"USERNAME": self.username, "PASSWORD": self.password}
        if self.client_secret:
            auth_parameters["SECRET_HASH"] = self._secret_hash(self.username)
        result = self._cognito_post(
            "InitiateAuth",
            {
                "AuthFlow": "USER_PASSWORD_AUTH",
                "AuthParameters": auth_parameters,
                "ClientId": self.client_id,
            },
        )
        return self._unpack(result)

    def _refresh_login(self, refresh_token):
        auth_parameters = {"REFRESH_TOKEN": refresh_token}
        if self.client_secret and self.username:
            auth_parameters["SECRET_HASH"] = self._secret_hash(self.username)
        result = self._cognito_post(
            "InitiateAuth",
            {
                "AuthFlow": "REFRESH_TOKEN_AUTH",
                "AuthParameters": auth_parameters,
                "ClientId": self.client_id,
            },
        )
        # A refresh response carries no new RefreshToken; the existing one stays
        # valid until the pool's refresh-token validity runs out.
        return self._unpack(result, keep_refresh_token=refresh_token)

    def _password_verifier_replay(self):
        """Replays a PASSWORD_VERIFIER challenge captured by hand into .env.

        Kept because it is the exact call the browser makes, but the claim
        signature is bound to its TIMESTAMP: this works for a few minutes after
        capture and then fails. Use the srp flow for anything long-running.
        """
        challenge_responses = {
            "USERNAME": _env("COGNITO_USERNAME"),
            "PASSWORD_CLAIM_SECRET_BLOCK": _env("COGNITO_PASSWORD_CLAIM_SECRET_BLOCK"),
            "TIMESTAMP": _env("COGNITO_PASSWORD_CLAIM_TIMESTAMP"),
            "PASSWORD_CLAIM_SIGNATURE": _env("COGNITO_PASSWORD_CLAIM_SIGNATURE"),
        }
        missing = [k for k, v in challenge_responses.items() if not v]
        if missing:
            raise AuthError(f"password_verifier flow is missing .env values: {missing}")

        result = self._cognito_post(
            "RespondToAuthChallenge",
            {
                "ChallengeName": "PASSWORD_VERIFIER",
                "ChallengeResponses": challenge_responses,
                "ClientId": self.client_id,
            },
        )
        return self._unpack(result)

    def _unpack(self, result, keep_refresh_token=None):
        auth_result = result.get("AuthenticationResult")
        if not auth_result:
            raise AuthError(f"No AuthenticationResult in response: {str(result)[:300]}")

        token = auth_result.get("IdToken" if self.token_kind == "id" else "AccessToken")
        if not token:
            raise AuthError(f"No {self.token_kind} token in AuthenticationResult")

        self._refresh_token = auth_result.get("RefreshToken") or keep_refresh_token or self._refresh_token
        expires_in = int(auth_result.get("ExpiresIn", 3600))
        return token, expires_in

    # --- public API ---------------------------------------------------------

    def _acquire(self):
        """Gets a token by whichever flow is configured. Caller holds the lock."""
        if self.flow == "static":
            if not self._static_token:
                raise AuthError("COGNITO_ACCESS_TOKEN is empty")
            # A pasted token carries no expiry we can trust; assume the Cognito
            # default hour so it is at least re-read from .env eventually.
            return self._static_token, 3600

        if not self.client_id:
            raise AuthError("COGNITO_CLIENT_ID is not set")

        # A refresh is far cheaper than a fresh handshake, so it is tried first
        # whenever a refresh token is on hand - regardless of the login flow.
        if self._refresh_token:
            try:
                return self._refresh_login(self._refresh_token)
            except AuthError as e:
                if self.flow == "refresh_token":
                    raise
                print(f"[auth] refresh failed ({e}); falling back to a full {self.flow} login")
                self._refresh_token = None

        if self.flow == "srp":
            return self._srp_login()
        if self.flow == "password":
            return self._password_login()
        if self.flow == "password_verifier":
            return self._password_verifier_replay()
        if self.flow == "refresh_token":
            raise AuthError("COGNITO_REFRESH_TOKEN is not set or no longer valid")
        raise AuthError(f"Unsupported AUTH_FLOW: {self.flow}")

    def token(self, force_refresh=False):
        """Returns a valid access token, or None when auth is switched off.

        Raises AuthError if auth is configured but the token cannot be obtained -
        callers decide whether that is fatal or worth retrying on the next tick.
        """
        if not self.enabled:
            return None

        with self._lock:
            if not force_refresh and self._token and time.time() < self._expires_at:
                return self._token

            try:
                token, expires_in = self._acquire()
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                raise

            self._token = token
            self._expires_at = time.time() + max(0, expires_in - REFRESH_MARGIN_SECONDS)
            self.last_error = None
            self.last_refresh_at = time.time()
            self.refresh_count += 1
            print(
                f"[auth] token acquired via {self.flow} "
                f"(valid {expires_in}s, renewing in {expires_in - REFRESH_MARGIN_SECONDS}s)"
            )
            return self._token

    def auth_header(self, force_refresh=False):
        """`{"Authorization": "Bearer ..."}`, or `{}` when auth is off/unavailable.

        Never raises: an API that happens to accept unauthenticated calls keeps
        working even if the credentials are wrong, and the failure is reported
        through `status()` instead of taking the poller down.
        """
        try:
            token = self.token(force_refresh=force_refresh)
        except Exception:
            return {}
        return {"Authorization": f"Bearer {token}"} if token else {}

    def invalidate(self):
        """Drops the cached token so the next call re-authenticates (used on a 401)."""
        with self._lock:
            self._token = None
            self._expires_at = 0.0

    def status(self):
        """Non-secret view of the auth state, safe to expose over HTTP."""
        with self._lock:
            seconds_left = int(self._expires_at - time.time()) if self._token else None
            return {
                "enabled": self.enabled,
                "flow": self.flow,
                "missing_config": self.missing_config,
                "token_kind": self.token_kind,
                "has_token": bool(self._token),
                "seconds_until_refresh": max(0, seconds_left) if seconds_left is not None else None,
                "refresh_count": self.refresh_count,
                "has_refresh_token": bool(self._refresh_token),
                "last_error": self.last_error,
            }


# One shared provider: a single token, refreshed once, used by every caller.
auth = CognitoAuth()


def authed_get(session, url, apply_auth=True, **kwargs):
    """GET with the Bearer token attached, retrying once on a 401/403.

    A token can be rejected mid-flight (pool restart, revoked session, clock
    skew), and the retry - with a forced fresh token - turns that into a blip
    rather than a gap in the data.
    """
    if not apply_auth or not auth.enabled:
        return session.get(url, **kwargs)

    headers = dict(kwargs.pop("headers", {}) or {})
    headers.update(auth.auth_header())
    response = session.get(url, headers=headers, **kwargs)

    if response.status_code in (401, 403):
        auth.invalidate()
        retry_headers = dict(headers)
        retry_headers.update(auth.auth_header(force_refresh=True))
        if "Authorization" in retry_headers:
            response = session.get(url, headers=retry_headers, **kwargs)

    return response


if __name__ == "__main__":
    import json

    print(json.dumps(auth.status(), indent=2))
    if auth.enabled:
        try:
            token = auth.token()
            print(f"\nToken ({len(token)} chars):\n{token}")
        except Exception as e:
            print(f"\nToken FAILED: {e}")
    elif auth.missing_config:
        print(f"\nAuth is off: AUTH_FLOW={auth.flow} still needs {auth.missing_config} in .env.")
    else:
        print("\nAuth is disabled (AUTH_FLOW=none / nothing configured in .env).")
