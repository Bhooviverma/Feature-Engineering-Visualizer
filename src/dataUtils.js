import Papa from 'papaparse';

let cachedData = null;

/**
 * Loads the Titanic train.csv file from /public/train.csv
 */
export async function loadTitanicDataset() {
  if (cachedData) return cachedData;

  return new Promise((resolve, reject) => {
    Papa.parse('/train.csv', {
      download: true,
      header: true,
      dynamicTyping: false,
      skipEmptyLines: true,
      complete: (results) => {
        if (!results.data || results.data.length === 0) {
          reject(new Error('CSV file is empty or could not be loaded from /train.csv'));
          return;
        }
        cachedData = results.data;
        resolve(results.data);
      },
      error: (err) => {
        reject(err);
      }
    });
  });
}

// 1. Median calculation
export function calculateMedian(values) {
  const filtered = values
    .filter(v => v !== null && v !== undefined && !isNaN(v) && v !== '')
    .map(Number)
    .sort((a, b) => a - b);

  if (filtered.length === 0) return 0;
  const mid = Math.floor(filtered.length / 2);
  return filtered.length % 2 !== 0
    ? filtered[mid]
    : (filtered[mid - 1] + filtered[mid]) / 2;
}

// 2. Mode calculation
export function calculateMode(values) {
  const freq = {};
  let maxCount = 0;
  let mode = null;

  values.forEach(val => {
    if (val !== null && val !== undefined && val !== '') {
      freq[val] = (freq[val] || 0) + 1;
      if (freq[val] > maxCount) {
        maxCount = freq[val];
        mode = val;
      }
    }
  });
  return mode;
}

// 3. Min-Max Scaling
export function minMaxScale(val, min, max) {
  if (max === min) return 0;
  return (val - min) / (max - min);
}

// 4. Standardization (Z-score)
export function standardize(val, mean, std) {
  if (std === 0) return 0;
  return (val - mean) / std;
}

// 5. Mean & Standard Deviation
export function calculateStats(values) {
  const nums = values
    .filter(v => v !== null && v !== undefined && !isNaN(v) && v !== '')
    .map(Number);

  if (nums.length === 0) return { mean: 0, std: 1, min: 0, max: 1, count: 0 };

  const min = Math.min(...nums);
  const max = Math.max(...nums);
  const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
  const variance = nums.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / nums.length;
  const std = Math.sqrt(variance) || 1;

  return { mean, std, min, max, count: nums.length };
}

// 6. Fisher-Pearson Skewness coefficient
export function calculateSkewness(values) {
  const nums = values
    .filter(v => v !== null && v !== undefined && !isNaN(v) && v !== '')
    .map(Number);
  const n = nums.length;
  if (n < 3) return 0;

  const mean = nums.reduce((a, b) => a + b, 0) / n;
  const m2 = nums.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / n;
  const m3 = nums.reduce((sum, x) => sum + Math.pow(x - mean, 3), 0) / n;

  const std = Math.sqrt(m2);
  if (std === 0) return 0;

  return m3 / Math.pow(std, 3);
}

// 7. Regular expression passenger title extraction
export function extractTitle(fullName) {
  if (!fullName) return 'Other';
  const match = fullName.match(/,\s*([^\.]+)\./);
  if (match && match[1]) {
    const rawTitle = match[1].trim();
    if (['Mr', 'Mrs', 'Miss', 'Master'].includes(rawTitle)) {
      return rawTitle;
    }
    return 'Other';
  }
  return 'Other';
}

// 8. Histogram Bin Generator
export function buildHistogram(values, numBins = 10, isLog = false) {
  const nums = values
    .filter(v => v !== null && v !== undefined && !isNaN(v) && v !== '')
    .map(Number)
    .map(v => (isLog ? Math.log(1 + Math.max(0, v)) : v));

  if (nums.length === 0) return [];

  const min = Math.min(...nums);
  const max = Math.max(...nums);
  const step = (max - min) / numBins || 1;

  const bins = Array.from({ length: numBins }, (_, i) => ({
    binLabel: `${(min + i * step).toFixed(1)} - ${(min + (i + 1) * step).toFixed(1)}`,
    count: 0
  }));

  nums.forEach(v => {
    let index = Math.floor((v - min) / step);
    if (index >= numBins) index = numBins - 1;
    if (index >= 0) bins[index].count += 1;
  });

  return bins;
}
