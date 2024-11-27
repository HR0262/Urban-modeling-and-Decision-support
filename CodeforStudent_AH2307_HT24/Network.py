#import pandas
#import geopandas
#import numpy as np
from haversine import haversine
import networkx as nx

# A directed graph is created with nodes and links
# the link attributes inludes distance (the haversine distance is multiplied with 1.5 to approximate the road distance by car ), travel time, capactivity, total cost and volumn. 
def RoadNetwork(zones):
  G = nx.DiGraph()
  G.add_nodes_from(zones.columns)
  G.add_edge('centerN', 'centerE', distance=1.5 * haversine(zones['centerN']['centroid'], zones['centerE']['centroid']),
             traveltime=1, cap=2000, cost=0, volume=0)
  G.add_edge('centerN', 'centerS', distance=1.5 * haversine(zones['centerN']['centroid'], zones['centerS']['centroid']),
             traveltime=1, cap=2000, cost=0,
             volume=0)
  G.add_edge('centerN', 'centerW', distance=1.5 * haversine(zones['centerN']['centroid'], zones['centerW']['centroid']),
             traveltime=1, cap=2000, cost=0,
             volume=0)
  G.add_edge('centerN', 'NE', distance=1.5 * haversine(zones['centerN']['centroid'], zones['NE']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('centerN', 'N', distance=1.5 * haversine(zones['centerN']['centroid'], zones['N']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('centerE', 'E', distance=1.5 * haversine(zones['centerE']['centroid'], zones['E']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('centerE', 'NE', distance=1.5 * haversine(zones['centerE']['centroid'], zones['NE']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('centerW', 'N', distance=1.5 * haversine(zones['centerW']['centroid'], zones['N']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('centerW', 'NW', distance=1.5 * haversine(zones['centerW']['centroid'], zones['NW']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('centerW', 'SW', distance=1.5 * haversine(zones['centerW']['centroid'], zones['SW']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('centerS', 'SW', distance=1.5 * haversine(zones['centerS']['centroid'], zones['SW']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('centerS', 'S', distance=1.5 * haversine(zones['centerS']['centroid'], zones['S']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('centerS', 'SE', distance=1.5 * haversine(zones['centerS']['centroid'], zones['SE']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('NE', 'N', distance=1.5 * haversine(zones['NE']['centroid'], zones['N']['centroid']), traveltime=1, cap=2000,
             cost=0,  volume=0)
  G.add_edge('NW', 'N', distance=1.5 * haversine(zones['NW']['centroid'], zones['N']['centroid']), traveltime=1, cap=2000,
             cost=0,  volume=0)

  G.add_edge('centerE', 'centerN', distance=1.5 *haversine(zones['centerE']['centroid'], zones['centerN']['centroid']),
             traveltime=1, cap=2000, cost=0, volume=0)
  G.add_edge('centerS', 'centerN', distance=1.5 *haversine(zones['centerS']['centroid'], zones['centerN']['centroid']),
             traveltime=1, cap=2000, cost=0,
             volume=0)
  G.add_edge('centerW', 'centerN', distance=1.5 *haversine(zones['centerW']['centroid'], zones['centerN']['centroid']),
             traveltime=1, cap=2000, cost=0,
             volume=0)
  G.add_edge('NE', 'centerN', distance=1.5 * haversine(zones['NE']['centroid'], zones['centerN']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('N', 'centerN', distance=1.5 * haversine(zones['N']['centroid'], zones['centerN']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('E', 'centerE', distance=haversine(zones['E']['centroid'], zones['centerE']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('NE', 'centerE', distance=1.5 * haversine(zones['NE']['centroid'], zones['centerE']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('N', 'centerW', distance=1.5 * haversine(zones['N']['centroid'], zones['centerW']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('NW', 'centerW', distance=1.5 * haversine(zones['NW']['centroid'], zones['centerW']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('SW', 'centerW', distance=1.5 * haversine(zones['SW']['centroid'], zones['centerW']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('SW', 'centerS', distance=1.5 * haversine(zones['SW']['centroid'], zones['centerS']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('S', 'centerS', distance=1.5 * haversine(zones['S']['centroid'], zones['centerS']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('SE', 'centerS', distance=1.5 * haversine(zones['SE']['centroid'], zones['centerS']['centroid']), traveltime=1,
             cap=2000, cost=0,
             volume=0)
  G.add_edge('N', 'NE', distance=1.5 * haversine(zones['N']['centroid'], zones['NE']['centroid']), traveltime=1, cap=2000,
             cost=0, volume=0)
  G.add_edge('N', 'NW', distance=1.5 * haversine(zones['N']['centroid'], zones['NW']['centroid']), traveltime=1, cap=2000,
             cost=0, volume=0)
  return (G)

def TransitNetwork(zones):
  G = nx.DiGraph()
  G.add_nodes_from(zones.columns)
  G.add_edge('NE', 'centerE', inv_time=10, wait_time=5, volume=0)   #T-bana danderyd-östermalm
  G.add_edge('centerE', 'NE', inv_time=10, wait_time=5, volume=0)  # T-bana danderyd-östermalm
  G.add_edge('centerE', 'E', inv_time=15, wait_time=8, volume=0)  # Tram östermalm-lidingö
  G.add_edge('E', 'centerE', inv_time=15, wait_time=8, volume=0)  # Tram östermalm-lidingö
  G.add_edge('centerN', 'centerE', inv_time=5, wait_time=2, volume=0)  #T-bana östermalm-centralen
  G.add_edge('centerE', 'centerN', inv_time=5, wait_time=2, volume=0)  # T-bana östermalm-centralen
  G.add_edge('centerN', 'centerS', inv_time=6, wait_time=2, volume=0)  # T-bana södermalm-centralen
  G.add_edge('centerS', 'centerN', inv_time=6, wait_time=2, volume=0)  # T-bana södermalm-centralen
  G.add_edge('centerS', 'S', inv_time=15, wait_time=4, volume=0)  # T-bana södermalm-south sthlm
  G.add_edge('S', 'centerS', inv_time=15, wait_time=4, volume=0)  # T-bana södermalm-south sthlm
  G.add_edge('centerS', 'SW', inv_time=15, wait_time=4, volume=0)  # T-bana södermalm-southwest sthlm
  G.add_edge('SW', 'centerS', inv_time=15, wait_time=4, volume=0)  # T-bana södermalm-southwest sthlm
  G.add_edge('centerS', 'SE', inv_time=15, wait_time=8, volume=0)  # Tram södermalm-nacka
  G.add_edge('SE', 'centerS', inv_time=15, wait_time=8, volume=0)  # Tram södermalm-nacka
  G.add_edge('N', 'centerW', inv_time=10, wait_time=5, volume=0)  # T-bana kungsholm-north sthlm
  G.add_edge('centerW', 'N', inv_time=10, wait_time=5, volume=0)  # T-bana kungsholm-north sthlm
  G.add_edge('centerW', 'centerN', inv_time=8, wait_time=4, volume=0)  # T-bana kungsholm-centralen
  G.add_edge('centerN', 'centerW', inv_time=8, wait_time=4, volume=0)  # T-bana kungsholm-centralen
  G.add_edge('NW', 'centerW', inv_time=15, wait_time=8, volume=0)  # T-bana kungsholm-northwest sthlm
  G.add_edge('centerW', 'NW', inv_time=15, wait_time=8, volume=0)  # T-bana kungsholm-northwest sthlm
  return G