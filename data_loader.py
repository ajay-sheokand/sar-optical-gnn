import ee 

def initialize_earth_engine(project_id):
    """
    Initializes the Google Earth Engine API with your specific project.
    """
    try:
        # Correct parameter for initialization is 'project'
        ee.Initialize(project=project_id)
        print(f"Google Earth Engine initialized successfully with project: {project_id}")
    except Exception as e:
        print("Initialization failed, starting authentication...")
        ee.Authenticate()
        ee.Initialize(project=project_id)

def get_delhi_roi():
    """ 
    Returns the region of interest for Delhi, India as an ee.Geometry.Rectangle.
    """
    return ee.Geometry.Rectangle([76.8, 28.4, 77.4, 28.9])

def mask_s2_clouds(img):
    """ 
    Masks the clouds and shadows using the s2cloudless dataset.
    """
    cld_prb = ee.Image(img.get('cloud_mask')).select('probability')
    is_cloud = cld_prb.gt(50)  # Threshold
    return img.updateMask(is_cloud.Not())

def get_sar_optical_data(roi, start_date, end_date):
    """
    Fetches, masks, and aligns Sentinel-1 and Sentinel-2 data.
    """
    # 1. Fetch Sentinel-1 (SAR)
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
           .filterBounds(roi) \
           .filterDate(start_date, end_date) \
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    
    # 2. Fetch Sentinel-2 (Optical) and join with cloud probability
    s2_sr = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate(start_date, end_date)
    s2_clouds = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(roi).filterDate(start_date, end_date)

    # Join collections on system:index to link probability to SR images
    joined = ee.Join.saveFirst('cloud_mask').apply(
        primary=s2_sr,
        secondary=s2_clouds,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
    )

    # 3. Apply Cloud Masking and Align Projection
    def align_and_mask(img):
        # Reproject to ensure consistent pixel grid
        return img.resample('bilinear').reproject(crs='EPSG:4326', scale=10)

    s2_masked = ee.ImageCollection(joined).map(mask_s2_clouds).map(align_and_mask)
    s1_aligned = s1.map(align_and_mask)

    return s1_aligned, s2_masked

if __name__ == "__main__":
    # Replace 'your-gcp-project-id' with your actual Google Cloud Project ID
    PROJECT_ID = 'sar-optical-gnn' 
    
    # Initialize
    initialize_earth_engine(PROJECT_ID)
    
    # Fetch Data
    roi = get_delhi_roi()
    s1, s2 = get_sar_optical_data(roi, '2020-01-01', '2020-01-30')
    
    # Print results
    print(f"Data ready: S1 image count = {s1.size().getInfo()}")
    print(f"Data ready: S2 image count = {s2.size().getInfo()}")