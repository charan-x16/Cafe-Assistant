"""Username/password account identity for remembered cafe profiles.

Accounts are tenant-scoped customer identities. A username that exists for one
cafe does not identify the same customer at another cafe unless that tenant also
has an account row. Passwords are never stored directly; the module stores a
versioned PBKDF2-HMAC hash with a random salt and returns opaque bearer tokens
through the existing token table after successful registration or login.
"""

from __future__ import annotations

import base64
import hashlib
import re
import secrets
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cafe_assistant.db.models import Customer, CustomerProfile
from cafe_assistant.db.repositories.profile_repo import ensure_customer_profile

_PASSWORD_HASH_VERSION = "pbkdf2_sha256"
_PASSWORD_ITERATIONS = 390_000
_USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.@-]{2,99}$")


class AccountIdentityError(ValueError):
    """Raised when an account operation cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class AccountCredentials:
    """Verified tenant-scoped account credentials.

    Attributes:
        customer_id (int):
            Durable customer ID linked to the account.
        tenant_id (int):
            Tenant that owns the account row.
        username (str):
            Normalized username stored for login.
    """

    customer_id: int
    tenant_id: int
    username: str


async def create_customer_account(
    session: AsyncSession,
    *,
    tenant_id: int,
    username: str,
    password: str,
) -> AccountCredentials:
    """Create a tenant-scoped customer account with a password hash.

    Args:
        session (AsyncSession):
            Async database session used for account and profile writes.
        tenant_id (int):
            Tenant that owns the username/password account.
        username (str):
            Customer-supplied username or email-like identifier.
        password (str):
            Raw password supplied during registration. It is validated and then
            hashed before any database write.

    Returns:
        AccountCredentials:
            Durable customer identity created for the account.

    Raises:
        AccountIdentityError:
            Raised when the username/password is invalid or already registered
            within the tenant.
    """
    normalized_username = normalize_username(username)
    _validate_password(password)
    existing = await _customer_by_username(
        session,
        tenant_id=tenant_id,
        username=normalized_username,
    )
    if existing is not None:
        raise AccountIdentityError("Username is already registered for this cafe.")

    customer = Customer(
        tenant_id=tenant_id,
        username=normalized_username,
        password_hash=hash_password(password),
    )
    customer.profile = CustomerProfile(preferences={}, dietary_facts={})
    session.add(customer)
    await session.flush()
    return AccountCredentials(
        customer_id=customer.id,
        tenant_id=tenant_id,
        username=normalized_username,
    )


async def authenticate_customer_account(
    session: AsyncSession,
    *,
    tenant_id: int,
    username: str,
    password: str,
) -> AccountCredentials | None:
    """Verify username/password credentials inside one tenant.

    Args:
        session (AsyncSession):
            Async database session used to load the account row.
        tenant_id (int):
            Tenant that must own the account.
        username (str):
            Customer-supplied username or email-like identifier.
        password (str):
            Raw password supplied during login.

    Returns:
        AccountCredentials | None:
            Verified durable identity when credentials are correct, otherwise
            None without revealing whether the username exists.
    """
    normalized_username = normalize_username(username)
    customer = await _customer_by_username(
        session,
        tenant_id=tenant_id,
        username=normalized_username,
    )
    if customer is None or not customer.password_hash:
        return None
    if not verify_password(password, customer.password_hash):
        return None
    await ensure_customer_profile(session, customer)
    return AccountCredentials(
        customer_id=customer.id,
        tenant_id=tenant_id,
        username=normalized_username,
    )


def normalize_username(username: str) -> str:
    """Normalize and validate an account username.

    Args:
        username (str):
            Customer-supplied username or email-like identifier.

    Returns:
        str:
            Lowercase, stripped username safe to compare within a tenant.

    Raises:
        AccountIdentityError:
            Raised when the normalized username is too short, too long, or
            contains unsupported characters.
    """
    normalized = username.strip().lower()
    if not _USERNAME_PATTERN.fullmatch(normalized):
        raise AccountIdentityError(
            "Username must be 3-100 characters using letters, numbers, dot, dash, underscore, or @."
        )
    return normalized


def hash_password(password: str) -> str:
    """Hash a raw password using a versioned PBKDF2-HMAC format.

    Args:
        password (str):
            Raw password supplied by the customer.

    Returns:
        str:
            Encoded hash string containing algorithm, iterations, salt, and
            derived key. This is the only password value stored in the database.
    """
    _validate_password(password)
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PASSWORD_ITERATIONS,
    )
    return "$".join(
        [
            _PASSWORD_HASH_VERSION,
            str(_PASSWORD_ITERATIONS),
            base64.urlsafe_b64encode(salt).decode("ascii"),
            base64.urlsafe_b64encode(digest).decode("ascii"),
        ]
    )


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a raw password against a stored PBKDF2 hash.

    Args:
        password (str):
            Raw password supplied by the customer during login.
        stored_hash (str):
            Stored versioned password hash from the customer account row.

    Returns:
        bool:
            True when the password matches the stored hash, otherwise False.
    """
    try:
        version, iterations_text, salt_text, digest_text = stored_hash.split("$", 3)
        if version != _PASSWORD_HASH_VERSION:
            return False
        iterations = int(iterations_text)
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return secrets.compare_digest(actual, expected)


def _validate_password(password: str) -> None:
    """Validate password length before hashing or authentication.

    Args:
        password (str):
            Raw password supplied by the customer.

    Returns:
        None:
            The function returns only when the password meets length bounds.

    Raises:
        AccountIdentityError:
            Raised when the password is outside the accepted range.
    """
    if len(password) < 8 or len(password) > 128:
        raise AccountIdentityError("Password must be between 8 and 128 characters.")


async def _customer_by_username(
    session: AsyncSession,
    *,
    tenant_id: int,
    username: str,
) -> Customer | None:
    """Load one account row by tenant and normalized username.

    Args:
        session (AsyncSession):
            Async database session used for the lookup.
        tenant_id (int):
            Tenant that owns the account namespace.
        username (str):
            Normalized username to match.

    Returns:
        Customer | None:
            Customer with profile eagerly loaded, or None when absent.
    """
    return await session.scalar(
        select(Customer)
        .where(Customer.tenant_id == tenant_id, Customer.username == username)
        .options(selectinload(Customer.profile))
    )