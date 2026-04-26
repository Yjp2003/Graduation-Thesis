"""
Supabase client initialization.
Provides both an admin client (service role) and a function to create
per-user clients using their JWT token.
"""
import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def get_supabase_admin() -> Client:
    """
    Admin client using SERVICE_ROLE_KEY.
    Bypasses RLS - use only for admin operations (user management).
    """
    if not SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SERVICE_ROLE_KEY == "your_service_role_key_here":
        # Fallback to anon key if service role not configured
        return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def get_supabase_client() -> Client:
    """
    Standard client using ANON_KEY.
    Respects RLS policies. Used for most operations.
    """
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
