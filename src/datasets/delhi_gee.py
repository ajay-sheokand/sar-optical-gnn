"""
Delhi ROI SAR/optical fetch + export, via Google Earth Engine.

Role in this project (docs/RESEARCH_PLAN.md §4, "Role of the existing Delhi/GEE pipeline"): this
is NOT a training data source. It's the out-of-distribution qualitative demo for M7 — monsoon-
season Delhi has usable SAR nearly continuously but usable cloud-free optical only in narrow
windows, which is the exact real-world gap this whole project's SAR-to-optical translation model
is meant to help with. There's no paired ground-truth optical for the cloudy periods by
construction, so this can only ever be a qualitative "does the trained model produce something
sensible here" check, never a source of (SAR, optical) training pairs.

What changed from the original version of this file: it fetched SAR/optical ImageCollections and
only ever printed how many images matched the filter (`.size().getInfo()`) — nothing was ever
written anywhere. `export_image_to_drive()` below is new, and actually starts a real Earth Engine
batch export task.

IMPORTANT — this file's export functions are NOT run as part of setting this file up. Calling
`ee.batch.Export...().start()` creates a real, asynchronous task against *your* Google Cloud
project (the PROJECT_ID below), consumes your Earth Engine quota, and needs `ee.Authenticate()`
(an interactive, browser-based login) the first time it runs on a machine. That's a deliberate,
account-specific action for you to trigger yourself when you're ready to actually pull Delhi
imagery — not something to run silently as part of a docs/tests pass.
"""

import ee


def initialize_earth_engine(project_id):
    """
    Initializes the Google Earth Engine API with your specific project.
    """
    try:
        # Correct parameter for initialization is 'project'
        ee.Initialize(project=project_id)
        print(f"Google Earth Engine initialized successfully with project: {project_id}")
    except Exception:
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

    Returns two ee.ImageCollections (SAR, optical) -- these are still just *filtered
    collections*, not a single exportable image. See build_export_composites() below for turning
    each collection into one ee.Image, which is what Earth Engine's export API actually needs.
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


def build_export_composites(s1_collection, s2_collection, roi):
    """
    Reduce each filtered ImageCollection down to one exportable ee.Image, clipped to `roi`.

    Earth Engine's export API (ee.batch.Export.image.*) exports a single ee.Image, not a
    collection -- a monsoon-season date range can match many S1/S2 passes, so this takes the
    per-pixel median across whatever images matched the filter. A median composite is a
    reasonable choice specifically for a *qualitative* demo (visually representative, robust to
    any single noisy/cloudy pass slipping through the mask) -- it would be the wrong choice if
    this were feeding model training, where you'd want individual real acquisitions, not a
    blended composite. That's consistent with this file's role: qualitative demo only (see the
    module docstring).
    """
    s1_composite = s1_collection.select('VV').median().clip(roi)
    s2_composite = s2_collection.select(['B4', 'B3', 'B2']).median().clip(roi)  # true-color RGB
    return s1_composite, s2_composite


def export_image_to_drive(image, description, folder, region, scale=10, crs="EPSG:4326", max_pixels=1e9):
    """
    Start an asynchronous Earth Engine export task that writes `image` to a GeoTIFF in your
    Google Drive.

    This function only *starts* the task (`.start()`) -- it does not wait for it to finish.
    Exports typically take anywhere from under a minute to tens of minutes depending on region
    size and server load. Progress can be checked at https://code.earthengine.google.com/tasks,
    or programmatically via `task.status()` on the object this function returns.

    Args:
        image: a single ee.Image (e.g. from build_export_composites()) -- not a collection.
        description: task name, also used as the default output filename.
        folder: destination folder name within your Google Drive.
        region: an ee.Geometry (or GeoJSON-compatible coordinate list) bounding the export.
        scale: output pixel size in meters. 10m matches Sentinel-1/2's native resolution.
        crs: output coordinate reference system.
        max_pixels: Earth Engine safety cap on exported pixel count; raise this if the export
            fails with a "too many pixels" error for a larger ROI than Delhi's.

    Returns:
        the started ee.batch.Task object.
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=max_pixels,
    )
    task.start()
    print(f"Started export task '{description}' -> Drive folder '{folder}'. "
          f"Monitor at https://code.earthengine.google.com/tasks")
    return task


def export_delhi_sar_optical(project_id, start_date, end_date, drive_folder="sar-optical-gnn-delhi"):
    """
    End-to-end: initialize Earth Engine, fetch Delhi SAR/optical for the given date range, and
    start export tasks for both as GeoTIFFs in Google Drive.

    Not called automatically anywhere in this module -- see the module docstring for why this is
    a deliberate, manually-triggered action rather than something that runs on import or on a
    test pass. Call this yourself, e.g. from a notebook or `python -c`, when you're ready to
    actually pull Delhi imagery for the M7 qualitative demo.

    Returns:
        (s1_task, s2_task) -- the two started ee.batch.Task objects.
    """
    initialize_earth_engine(project_id)

    roi = get_delhi_roi()
    s1_collection, s2_collection = get_sar_optical_data(roi, start_date, end_date)
    s1_composite, s2_composite = build_export_composites(s1_collection, s2_collection, roi)

    s1_task = export_image_to_drive(
        s1_composite, description=f"delhi_s1_{start_date}_{end_date}", folder=drive_folder, region=roi
    )
    s2_task = export_image_to_drive(
        s2_composite, description=f"delhi_s2_{start_date}_{end_date}", folder=drive_folder, region=roi
    )
    return s1_task, s2_task


if __name__ == "__main__":
    # Replace 'your-gcp-project-id' with your actual Google Cloud Project ID
    PROJECT_ID = 'sar-optical-gnn'

    export_delhi_sar_optical(PROJECT_ID, '2020-01-01', '2020-01-30')
