// ============================================================
// ERA5-Land: Hourly Evaporation & Potential Evapotranspiration
// per HRU — Foux karst system
// Period: Sep 2019 – Jul 2024  |  Split by year to avoid timeout
// Output: 6 CSV files to Google Drive folder "GEE_Foux_Exports"
// ============================================================
// NOTE: ERA5-Land is HOURLY (not 15-min — no 15-min ERA5 exists).
// Values are in meters (negative = evaporation flux upward).
// We apply .abs().multiply(1000) to get positive mm/hour.
//
// Small HRU fix: bilinear resampling at 1 km scale interpolates
// ERA5-Land (~11 km pixels) so even sub-pixel HRUs get a value.
// ============================================================

var hrus = ee.FeatureCollection('projects/ee-armanjalil31/assets/foux-HRU');
print('HRUs loaded:', hrus.size());

// ERA5-Land HOURLY — select evaporation variables only
var ERA5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
  .select(['total_evaporation', 'potential_evaporation']);

// Study periods — one export task per period to avoid GEE timeout
var periods = [
  {start: '2019-09-01', end: '2020-01-01', label: '2019'},
  {start: '2020-01-01', end: '2021-01-01', label: '2020'},
  {start: '2021-01-01', end: '2022-01-01', label: '2021'},
  {start: '2022-01-01', end: '2023-01-01', label: '2022'},
  {start: '2023-01-01', end: '2024-01-01', label: '2023'},
  {start: '2024-01-01', end: '2024-08-01', label: '2024'},
];

function exportPeriod(startDate, endDate, label) {
  var collection = ERA5.filterDate(startDate, endDate);

  var fc = collection.map(function(image) {
    // Convert m to mm and flip sign (ERA5 evaporation is negative)
    var img = image.abs().multiply(1000);

    // Bilinear resampling: interpolates ERA5-Land ~11 km pixels to 1 km.
    // Ensures even HRUs smaller than one ERA5 pixel receive a value.
    img = img.resample('bilinear');

    var timestamp = ee.Date(image.get('system:time_start'))
                      .format('YYYY-MM-dd HH:mm:ss');

    var reduced = img.reduceRegions({
      collection: hrus,
      reducer: ee.Reducer.mean(),
      scale: 1000,
      tileScale: 4
    });

    return reduced.map(function(f) {
      return f.set('datetime', timestamp);
    });
  });

  var flat = ee.FeatureCollection(fc).flatten();

  Export.table.toDrive({
    collection: flat,
    description: 'ERA5_Evap_ETP_HRU_' + label,
    folder: 'GEE_Foux_Exports',
    fileNamePrefix: 'ERA5_Evap_ETP_HRU_' + label,
    fileFormat: 'CSV',
    selectors: ['datetime', 'HRU', 'total_evaporation', 'potential_evaporation']
  });

  print('Task queued: ERA5_Evap_ETP_HRU_' + label);
}

periods.forEach(function(p) {
  exportPeriod(p.start, p.end, p.label);
});

print('All 6 tasks created. Go to Tasks tab and click RUN on each.');
print('Output folder in Google Drive: GEE_Foux_Exports');
print('Columns: datetime | HRU | total_evaporation (mm/hr) | potential_evaporation (mm/hr)');