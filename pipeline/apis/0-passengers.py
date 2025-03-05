#!/usr/bin/env python3
"""
Returns a list of ships
"""


import requests


def availableShips(passengerCount):
    """
    Returns a list of ships
    that can hold a given number of passengers
    """
    # API endpoint
    url = "https://swapi-api.alx-tools.com/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            return []

        data = response.json()
        for ship in data['results']:
            if ship['passengers'] != "unknown":
                try:
                    if int(ship['passengers'].replace(',', '')) >= passengerCount:
                        ships.append(ship['name'])
                except ValueError:
                    pass

        url = data.get('next')

    return ships