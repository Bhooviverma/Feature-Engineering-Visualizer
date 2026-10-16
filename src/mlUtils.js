import { calculateMedian, calculateStats, standardize, extractTitle } from './dataUtils';

/**
 * Deterministic pseudo-random split
 */
function trainTestSplit(rows, testRatio = 0.25, seed = 1337) {
  let s = seed;
  const random = () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };

  const shuffled = [...rows].sort(() => random() - 0.5);
  const testCount = Math.floor(shuffled.length * testRatio);
  return {
    train: shuffled.slice(testCount),
    test: shuffled.slice(0, testCount)
  };
}

/**
 * Evaluates Logistic Regression on Titanic data with zero data leakage.
 * All parameters (median, scaling stats) are learned strictly on the training partition.
 */
export function runLogisticRegressionExperiment(rawRows) {
  const { train, test } = trainTestSplit(rawRows, 0.25, 1337);

  // Train parameters (learned ONLY from train split to prevent data leakage)
  const trainAges = train.map(r => r.Age).filter(v => v !== '' && !isNaN(v)).map(Number);
  const ageMedian = calculateMedian(trainAges);
  const trainFares = train.map(r => r.Fare).filter(v => v !== '' && !isNaN(v)).map(Number);
  const fareMedian = calculateMedian(trainFares);
  const fareLogValues = trainFares.map(f => Math.log(1 + f));
  const fareLogStats = calculateStats(fareLogValues);
  const ageStats = calculateStats(trainAges);

  // Model A: Basic representation [Bias, Pclass, Age_imputed, Fare_imputed]
  const extractFeaturesA = (row) => {
    const pclass = Number(row.Pclass) || 3;
    const age = Number(row.Age) || ageMedian;
    const fare = Number(row.Fare) || fareMedian;
    return [1, (pclass - 1) / 2, (age - 20) / 20, (fare - 30) / 50];
  };

  // Model B: Full engineered representation
  const extractFeaturesB = (row) => {
    const pclass = Number(row.Pclass) || 3;
    const pclassNorm = (pclass - 1) / 2;
    const age = Number(row.Age) || ageMedian;
    const ageStd = standardize(age, ageStats.mean, ageStats.std);
    const fare = Number(row.Fare) || fareMedian;
    const fareLog = Math.log(1 + Math.max(0, fare));
    const fareLogStd = standardize(fareLog, fareLogStats.mean, fareLogStats.std);

    const sexMale = row.Sex === 'male' ? 1 : 0;
    const embC = row.Embarked === 'C' ? 1 : 0;
    const embQ = row.Embarked === 'Q' ? 1 : 0;
    const embS = (row.Embarked === 'S' || !row.Embarked) ? 1 : 0;

    const sibsp = Number(row.SibSp) || 0;
    const parch = Number(row.Parch) || 0;
    const familySize = sibsp + parch + 1;
    const familySizeNorm = (familySize - 1) / 10;
    const isAlone = familySize === 1 ? 1 : 0;

    const title = extractTitle(row.Name);
    const titleMaster = title === 'Master' ? 1 : 0;
    const titleMiss = title === 'Miss' ? 1 : 0;
    const titleMr = title === 'Mr' ? 1 : 0;
    const titleMrs = title === 'Mrs' ? 1 : 0;

    return [
      1,
      pclassNorm,
      ageStd,
      fareLogStd,
      sexMale,
      embC,
      embQ,
      embS,
      familySizeNorm,
      isAlone,
      titleMaster,
      titleMiss,
      titleMr,
      titleMrs
    ];
  };

  const sigmoid = (z) => 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, z))));

  // Gradient Descent Logistic Regression Trainer
  const trainLR = (trainData, featureExtractor, epochs = 180, lr = 0.15) => {
    const numFeatures = featureExtractor(trainData[0]).length;
    let weights = new Array(numFeatures).fill(0);

    for (let e = 0; e < epochs; e++) {
      const gradients = new Array(numFeatures).fill(0);
      for (const row of trainData) {
        const y = Number(row.Survived) || 0;
        const x = featureExtractor(row);
        let dot = 0;
        for (let i = 0; i < numFeatures; i++) dot += weights[i] * x[i];
        const pred = sigmoid(dot);
        const error = pred - y;
        for (let i = 0; i < numFeatures; i++) gradients[i] += error * x[i];
      }
      for (let i = 0; i < numFeatures; i++) {
        weights[i] -= (lr * gradients[i]) / trainData.length;
      }
    }
    return weights;
  };

  const evaluate = (testData, weights, featureExtractor) => {
    let tp = 0, fp = 0, tn = 0, fn = 0;

    testData.forEach(row => {
      const y = Number(row.Survived) || 0;
      const x = featureExtractor(row);
      let dot = 0;
      for (let i = 0; i < weights.length; i++) dot += weights[i] * x[i];
      const prob = sigmoid(dot);
      const pred = prob >= 0.5 ? 1 : 0;

      if (pred === 1 && y === 1) tp++;
      else if (pred === 1 && y === 0) fp++;
      else if (pred === 0 && y === 0) tn++;
      else if (pred === 0 && y === 1) fn++;
    });

    const total = tp + fp + tn + fn;
    const accuracy = total > 0 ? (tp + tn) / total : 0;
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    return { accuracy, precision, recall, f1, total, tp, fp, tn, fn };
  };

  const weightsA = trainLR(train, extractFeaturesA);
  const metricsA = evaluate(test, weightsA, extractFeaturesA);

  const weightsB = trainLR(train, extractFeaturesB);
  const metricsB = evaluate(test, weightsB, extractFeaturesB);

  return {
    trainSize: train.length,
    testSize: test.length,
    modelA: metricsA,
    modelB: metricsB
  };
}