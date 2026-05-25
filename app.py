"""Streamlit Cloud entry point.

This file is intentionally kept as a thin compatibility wrapper so the app can
be deployed with either `app.py` or the original `streamlit_app.py` as the main
file path.
"""

from streamlit_app import *  # noqa: F401,F403
