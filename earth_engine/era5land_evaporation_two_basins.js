// ==========================
// LOAD YOUR SHAPEFILES
// ==========================
var area1 = ee.FeatureCollection("projects/ee-armanjalil31/assets/bv_reppe");
var area2 = ee.FeatureCollection("users/armanjalil31/BV_hydrogeo_Foux");

// ==========================
// DISPLAY AREAS
// ==========================
Map.centerObject(area2, 10);

Map.addLayer(area1, {color: 'blue'}, 'Area 1 (bv_reppe)');
Map.addLayer(area2, {color: 'red'}, 'Area 2 (BV_hydrogeo_Foux)');

// ==========================
// LOAD ERA5-LAND DATA
// ==========================
var dataset = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                .select("total_evaporation");

// ==========================
// DEFINE TIME RANGE
// ==========================
var startDate = '2019-09-01';
var endDate   = '2020-08-01';

var era5 = dataset.filterDate(startDate, endDate);

// ==========================
// CONVERSION FUNCTION
// m → mm (and make positive)
// ==========================
var convertToMM = function(img) {
  return img.multiply(-1000) // convert to mm and flip sign
            .copyProperties(img, ['system:time_start']);
};

// Apply conversion
var evapMM = era5.map(convertToMM);

// ==========================
// FUNCTION TO COMPUTE MEAN PER AREA
// ==========================
var computeStats = function(featureCollection, name) {
  
  var stats = evapMM.map(function(img) {
    
    var mean = img.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: featureCollection.geometry(),
      scale: 9000,        // ERA5 resolution ~9 km
      maxPixels: 1e13,
      tileScale: 4        // improves stability
    });
    
    return ee.Feature(null, {
      'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd HH:mm'),
      'etp_mm_per_hour': mean.get('total_evaporation'),
      'area_name': name
    });
  });
  
  return stats;
};

// ==========================
// COMPUTE FOR BOTH AREAS
// ==========================
var stats1 = computeStats(area1, 'bv_reppe');
var stats2 = computeStats(area2, 'BV_hydrogeo_Foux');

// Merge results
var finalStats = stats1.merge(stats2);

// ==========================
// EXPORT TO GOOGLE DRIVE
// ==========================
Export.table.toDrive({
  collection: finalStats,
  description: 'ERA5_ETP_hourly_mm',
  fileFormat: 'CSV'
});