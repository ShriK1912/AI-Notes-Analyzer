"""
Central Firestore database helper.
All backend services must use get_db() from here — never call firestore.client() directly —
so the database name ('neww') is configured in one place.
"""
from __future__ import annotations
from firebase_admin import firestore as _firestore

# The project has a named Firestore database called 'neww'.
# The Firestore security rules are deployed only to this database.
# The default '(default)' database has no rules (deny-all).
FIRESTORE_DATABASE = "neww"


def get_db():
    """Return the Firestore client connected to the 'neww' named database."""
    return _firestore.client(database_id=FIRESTORE_DATABASE)


# Alias kept for any legacy callers
def get_database():
    return get_db()
