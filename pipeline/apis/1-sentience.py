#!/usr/bin/env python3
"""
returns the list of names of the home planets of all sentient species
"""


import requests


def sentientPlanets():
    """
    list of names of the home planets of all sentient species
    """
    url = "https://swapi-api.alx-tools.com/api/species/"

    planets = []

    while url:
        response = requests.get(url)
        data = response.json()
        for planet in data['results']:
            if planet['homeworld']!= "":
                planets.append(planet['name'])

        url = data.get('next')

    return planets