import rasterio
import numpy as np
from sklearn.cluster import KMeans
import folium
from pyproj import Transformer

def load_geotiff(file_path):
    with rasterio.open(file_path) as src:
        population_data = src.read(1) 
        transform = src.transform
        crs = src.crs
    return population_data, transform, crs

def find_population_centroid(population_data, transform):
    #getting coordinates of all populated areas
    y, x = np.where(population_data > 0)
    lats, lons = rasterio.transform.xy(transform, y, x)
    
    #weighing the coordinates by population density
    weights = population_data[population_data > 0]
    
    #implementing k means with k=1 to find the weighted centroid
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(np.column_stack((lats, lons)), sample_weight=weights)
    
    return kmeans.cluster_centers_[0]

def antipodal_point(lat, lon):
    return -lat, (lon + 180) % 360 - 180

def create_map(centroid_lat, centroid_lon, antipodal_lat, antipodal_lon):
    m = folium.Map(zoom_start=2)
    
    #centroid marker
    folium.Marker(
        [centroid_lat, centroid_lon],
        popup="Population Centroid",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)
    
    #antipodal marker
    folium.Marker(
        [antipodal_lat, antipodal_lon],
        popup="Furthest Point (Antipodal)",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    
    folium.PolyLine(
        locations=[[centroid_lat, centroid_lon], [antipodal_lat, antipodal_lon]],
        color="green",
        weight=2,
        opacity=0.8
    ).add_to(m)
    
    return m

def main(file_path):
   
    population_data, transform, crs = load_geotiff(file_path)
    
   
    centroid_lat, centroid_lon = find_population_centroid(population_data, transform)
    
    
    antipodal_lat, antipodal_lon = antipodal_point(centroid_lat, centroid_lon)
    
    print(f"Population Centroid: {centroid_lat:.4f}, {centroid_lon:.4f}")
    print(f"Furthest Point (Antipodal): {antipodal_lat:.4f}, {antipodal_lon:.4f}")
    
    
    m = create_map(centroid_lat, centroid_lon, antipodal_lat, antipodal_lon)
    m.save("global_population_analysis.html")
    print("Map saved as global_population_analysis.html")

if __name__ == "__main__":
    geotiff_file = '/Users/sunnydavid/Desktop/Dissertation COde/ppp_2020_1km_Aggregated.tif'
    main(geotiff_file)