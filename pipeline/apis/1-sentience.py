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

    planets = set()

    with requests.Session() as session:
        while url:
            response = session.get(url).json()

            for species in response['results']:
                if (
                    species.get('classification') == 'sentient' or
                    species.get('designation') == 'sentient'
                ):
                    homeworld_url = species.get('homeworld')
                    if homeworld_url:
                        planet_data = session.get(homeworld_url).json()
                        planets.add(planet_data.get('name', 'Unknown'))

            url = response.get('next')

    return planets
