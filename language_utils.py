"""Language utilities for the INTERGROWTH-21st Preterm Growth Tracker."""

import streamlit as st
from typing import Optional
import json


def get_browser_language() -> str:
    """Attempt to detect the user's browser language.
    
    Returns:
        str: The detected language code or 'en' if detection fails
    """
    # In Streamlit, we can't directly access browser headers
    # This is a placeholder that would need to be replaced with a proper implementation
    # when deployed with a framework that allows access to request headers
    return "en"  # Default to English


def save_language_preference(language_code: str) -> None:
    """Save the user's language preference in a cookie.
    
    Args:
        language_code: The language code to save
    """
    if 'language_preference' not in st.session_state:
        st.session_state.language_preference = language_code
    
    # Set cookie via JavaScript
    cookie_js = f"""
    <script>
        document.cookie = "language_preference={language_code}; path=/; max-age=31536000; SameSite=Lax";
    </script>
    """
    st.markdown(cookie_js, unsafe_allow_html=True)


def get_language_from_cookie() -> Optional[str]:
    """Get the user's language preference from cookie.
    
    Returns:
        Optional[str]: The language code from the cookie or None if not found
    """
    # In Streamlit, we can't directly access cookies
    # We'll use session state as a workaround
    return st.session_state.get('language_preference')


def show_cookie_consent() -> bool:
    """Show a cookie consent banner and return whether cookies are accepted.
    
    Returns:
        bool: True if cookies are accepted, False otherwise
    """
    from translations import get_translation
    
    # Get the current language code
    language_code = st.session_state.get('language_code', 'en')
    
    # Check if cookie consent has already been given
    if 'cookie_consent' not in st.session_state:
        st.session_state.cookie_consent = None
    
    # If consent hasn't been decided yet, show the banner
    if st.session_state.cookie_consent is None:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{get_translation('cookie_consent_title', language_code)}**")
                st.markdown(get_translation('cookie_consent_message', language_code))
            
            with col2:
                if st.button(get_translation('cookie_accept_button', language_code)):
                    st.session_state.cookie_consent = True
                    st.rerun()
            
            with col3:
                if st.button(get_translation('cookie_decline_button', language_code)):
                    st.session_state.cookie_consent = False
                    st.rerun()
    
    return st.session_state.cookie_consent == True


def initialize_language() -> str:
    """Initialize the language based on cookies or browser settings.
    
    Returns:
        str: The selected language code
    """
    from translations import LANGUAGE_OPTIONS
    
    # Initialize language code in session state if not present
    if 'language_code' not in st.session_state:
        # Try to get language from cookie first
        cookie_lang = get_language_from_cookie()
        if cookie_lang and cookie_lang in LANGUAGE_OPTIONS.values():
            st.session_state.language_code = cookie_lang
        else:
            # Try to detect browser language
            browser_lang = get_browser_language()
            # Map browser language to our supported languages
            if browser_lang.startswith('es'):
                st.session_state.language_code = 'es'
            elif browser_lang.startswith('pt'):
                st.session_state.language_code = 'pt_BR'
            else:
                st.session_state.language_code = 'en'  # Default to English
    
    return st.session_state.language_code