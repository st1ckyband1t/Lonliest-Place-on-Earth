import rasterio
import numpy as np
from scipy.spatial import Voronoi
import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPoint
import folium
from sklearn.cluster import MiniBatchKMeans

def load_geotiff(file_path):
    with rasterio.open(file_path) as src:
        population_data = src.read(1)
        transform = src.transform
        crs = src.crs
    return population_data, transform, crs

def downsample_population_points(population_data, transform, target_points=1000):
    #fetch coordinates of populated areas
    y, x = np.where(population_data > np.percentile(population_data, 90))  #filter to use top 10% populated areas
    lats, lons = rasterio.transform.xy(transform, y, x)
    
    #converting to array for clustering
    points = np.column_stack([lons, lats])
    
    # kmeans batching for memory-efficient clustering
    kmeans = MiniBatchKMeans(n_clusters=target_points, batch_size=1000)
    kmeans.fit(points)
    
    return kmeans.cluster_centers_

def process_voronoi_in_batches(points, batch_size=100):
    furthest_vertices = []
    
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        try:
            vor = Voronoi(batch_points)
            furthest_vertices.extend(vor.vertices.tolist())
        except Exception as e:
            print(f"Skipping batch due to error: {e}")
    
    return np.array(furthest_vertices)

def find_remote_point_voronoi(population_data, transform, world_gdf):
    #downsampling population points for efficiency
    downsampled_points = downsample_population_points(population_data, transform)
    
    #process Voronoi diagrams in batches
    vertices = process_voronoi_in_batches(downsampled_points)
    
    #converting vertices to GeoDataFrame
    vertices_gdf = gpd.GeoDataFrame(
        geometry=[Point(vertex) for vertex in vertices if not np.any(np.isnan(vertex))]
    )
    
    #filtering point to only those on land
    land_vertices = vertices_gdf[vertices_gdf.apply(
        lambda x: any(world_gdf.contains(x.geometry)), axis=1
    )]
    
    if land_vertices.empty:
        raise ValueError("No valid vertices found on land")
    
    #computing multipoint from population points for efficient distance calculation
    population_points = MultiPoint(downsampled_points)
    
    #compute vertex furthest from all populated points
    max_distance = 0
    furthest_point = None
    
    for vertex in land_vertices.geometry:
        dist = vertex.distance(population_points)
        if dist > max_distance:
            max_distance = dist
            furthest_point = vertex
    
    return furthest_point.y, furthest_point.x

def create_map(lat, lon, population_points):
    m = folium.Map(location=[lat, lon], zoom_start=4)
    
    #marker for furthest point
    folium.Marker(
        [lat, lon],
        popup="Furthest Point",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    #sample of population points
    for point in population_points[:100]:  #could be changed according to required points for visualization
        folium.CircleMarker(
            location=[point[1], point[0]],
            radius=1,
            color='blue',
            fill=True
        ).add_to(m)
    
    return m

def main(file_path):
    
    population_data, transform, crs = load_geotiff(file_path)
    
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
    try:
      
        lat, lon = find_remote_point_voronoi(population_data, transform, world)
        
      
        population_points = downsample_population_points(population_data, transform, target_points=100)
        
        #save map
        m = create_map(lat, lon, population_points)
        m.save("voronoi_population_analysis.html")
        
        print(f"Furthest point found at: {lat:.4f}, {lon:.4f}")
        print("Map saved as voronoi_population_analysis.html")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Consider using a different approach or further reducing the data size.")

if __name__ == "__main__":
    geotiff_file = '/Users/sunnydavid/Desktop/Dissertation COde/ppp_2020_1km_Aggregated.tif'
    main(geotiff_file)