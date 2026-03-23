"""
This module contains a list of proxy server addresses.
Please rename this file to proxies.py and add your working proxy list!
"""

import random

# Example dummy proxies (replace these with your own private proxies)
PROXIES = [
    "x.x.x.x:yyyy",
    "x.x.x.x:yyyy"
]

def get_random_proxy():
    """
    Returns a random proxy from the list.
    """
    return random.choice(PROXIES)
