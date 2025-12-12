import ipaddress
import os
import socket
from urllib.parse import urlparse


class UrlValidationError(ValueError):
    pass


def _is_ip_public(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def validate_outbound_url(
    url: str,
    *,
    allowed_domains: list[str] | None = None,
    allow_public_internet: bool = True,
    max_url_length: int = 2048,
) -> str:
    """
    SSRF guard for user-supplied URLs.
    - Only allows http/https
    - Blocks localhost/private/link-local/etc targets (incl. DNS that resolves to them)
    - Optional allowlist of domains (exact or subdomain)
    """
    if not isinstance(url, str) or not url.strip():
        raise UrlValidationError("Empty URL")

    url = url.strip()
    if len(url) > max_url_length:
        raise UrlValidationError("URL too long")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UrlValidationError("Only http/https URLs are allowed")
    if not parsed.netloc:
        raise UrlValidationError("Missing host")
    if parsed.username or parsed.password:
        raise UrlValidationError("Userinfo in URL is not allowed")

    host = (parsed.hostname or "").strip(".").lower()
    if not host:
        raise UrlValidationError("Missing host")

    if host in ("localhost",):
        raise UrlValidationError("Localhost is not allowed")

    # Domain allowlist (exact or subdomain)
    if allowed_domains is None:
        allowed_domains = _split_csv(os.getenv("ALLOWED_DOMAINS"))
    if allowed_domains:
        ok = any(host == d or host.endswith("." + d) for d in allowed_domains)
        if not ok:
            raise UrlValidationError("Host not in allowlist")

    # If it's an IP literal, validate it directly
    try:
        ipaddress.ip_address(host)
        is_ip_literal = True
    except ValueError:
        is_ip_literal = False

    if is_ip_literal:
        if not _is_ip_public(host):
            raise UrlValidationError("Private/non-public IPs are not allowed")
        if not allow_public_internet:
            raise UrlValidationError("Outbound internet is disabled")
        return url

    # Resolve DNS and block if any result is private/non-public
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as e:
        raise UrlValidationError(f"DNS resolution failed: {e}") from e

    resolved_ips: set[str] = set()
    for family, _, _, _, sockaddr in infos:
        if family == socket.AF_INET:
            resolved_ips.add(sockaddr[0])
        elif family == socket.AF_INET6:
            resolved_ips.add(sockaddr[0])

    if not resolved_ips:
        raise UrlValidationError("DNS resolution returned no addresses")

    for ip in resolved_ips:
        if not _is_ip_public(ip):
            raise UrlValidationError("Host resolves to private/non-public IP")

    if not allow_public_internet:
        raise UrlValidationError("Outbound internet is disabled")

    return url
