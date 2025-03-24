import rasterio
import numpy as np
from scipy.ndimage import distance_transform_edt
from pyproj import Geod
import folium

def load_population_data(file_path):
    with rasterio.open(file_path) as src:
        population = src.read(1)  
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    return population, transform, crs, nodata

def haversine_distance(lon1, lat1, lon2, lat2):
    geod = Geod(ellps="WGS84")
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    return distance / 1000  #converting to kilometers

def find_furthest_point(population, transform, nodata):
    #creating a binary mask where populated areas are 0 and unpopulated are 1
    mask = ((population == 0) | (population == nodata)).astype(np.uint8)
    
    #distance transform
    pixel_size_x = abs(transform[0])
    pixel_size_y = abs(transform[4])
    distances = distance_transform_edt(mask, sampling=[pixel_size_y, pixel_size_x])
    
    #position of the maximum distance
    max_distance_pos = np.unravel_index(np.argmax(distances), distances.shape)
    
    #convert to proper coordinates
    lon, lat = pixel_to_coordinates(max_distance_pos[1], max_distance_pos[0], transform)
    
    return lon, lat, distances[max_distance_pos]

def pixel_to_coordinates(col, row, transform):
    lon, lat = transform * (col, row)
    return lon, lat

def coordinates_to_pixel(lon, lat, transform):
    col, row = ~transform * (lon, lat)
    return int(col), int(row)

def create_map(furthest_point, point_nemo):
    #creates a map centered on the furthest point
    m = folium.Map(location=[furthest_point[1], furthest_point[0]], zoom_start=3)
    
    #adds a marker for the furthest point
    folium.Marker(
        [furthest_point[1], furthest_point[0]],
        popup="Furthest Point from Population",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)
    
    #adds a marker for Point Nemo
    folium.Marker(
        [point_nemo[1], point_nemo[0]],
        popup="Point Nemo (Known Oceanic Pole of Inaccessibility)",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)
    
    #connect the two markers
    folium.PolyLine(
        locations=[[furthest_point[1], furthest_point[0]], [point_nemo[1], point_nemo[0]]],
        color="green",
        weight=2,
        opacity=0.8,
    ).add_to(m)
    
    # Save the map
    m.save("furthest_point_map.html")

file_path = '/Users/sunnydavid/Desktop/Dissertation COde/ppp_2020_1km_Aggregated.tif'
population, transform, crs, nodata = load_population_data(file_path)

furthest_lon, furthest_lat, _ = find_furthest_point(population, transform, nodata)

print(f"The point furthest from human population is at longitude {furthest_lon:.4f}, latitude {furthest_lat:.4f}")

#calculating distance to neared populated are
rows, cols = population.shape
min_distance_km = float('inf')

for i in range(rows):
    for j in range(cols):
        if population[i, j] > 0 and population[i, j] != nodata:
            lon, lat = pixel_to_coordinates(j, i, transform)
            distance = haversine_distance(furthest_lon, furthest_lat, lon, lat)
            if distance < min_distance_km:
                min_distance_km = distance

print(f"The distance to the nearest populated area is approximately {min_distance_km:.2f} km")

#adding point nemo coordinates to compare
point_nemo = (-123.393333, -48.876667)

# creating map
create_map((furthest_lon, furthest_lat), point_nemo)

distance_to_nemo = haversine_distance(furthest_lon, furthest_lat, point_nemo[0], point_nemo[1])
print(f"Distance from our calculated point to Point Nemo: {distance_to_nemo:.2f} km")

point_nemo_col, point_nemo_row = coordinates_to_pixel(point_nemo[0], point_nemo[1], transform)
point_nemo_min_distance = float('inf')

for i in range(rows):
    for j in range(cols):
        if population[i, j] > 0 and population[i, j] != nodata:
            lon, lat = pixel_to_coordinates(j, i, transform)
            distance = haversine_distance(point_nemo[0], point_nemo[1], lon, lat)
            if distance < point_nemo_min_distance:
                point_nemo_min_distance = distance

print(f"Nearest population to Point Nemo: {point_nemo_min_distance:.2f} km")