"""Merge identity hardening and production catalog migration heads.

Revision ID: 20260622_0007
Revises: 20260618_0005, 20260619_0006
Create Date: 2026-06-22 00:00:00.000000

"""
from collections.abc import Sequence

revision: str = "20260622_0007"
down_revision: tuple[str, str] | None = ("20260618_0005", "20260619_0006")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Merge both migration branches without changing database objects.

    Args:
        None.

    Returns:
        None:
            Alembic records the merge revision after both parent revisions have
            run. No schema DDL is required for this merge point.
    """


def downgrade() -> None:
    """Move back from the merge point without changing database objects.

    Args:
        None.

    Returns:
        None:
            Alembic removes only the merge revision marker. Parent branch schema
            changes remain governed by their individual downgrade functions.
    """
