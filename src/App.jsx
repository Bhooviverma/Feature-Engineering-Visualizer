import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  loadTitanicDataset,
  calculateMedian,
  calculateMode,
  calculateStats,
  minMaxScale,
  standardize,
  calculateSkewness,
  extractTitle,
  buildHistogram
} from './dataUtils';
import { runLogisticRegressionExperiment } from './mlUtils';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import {
  ArrowLeft,
  ArrowRight,
  RotateCcw,
  Mic,
  ChevronRight,
  HelpCircle,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Database,
  Check,
  Play,
  GitFork,
  Plus,
  Filter,
  Activity,
  ShieldCheck,
  RefreshCw,
  Scissors
} from 'lucide-react';

/* =========================================================
   STAGE 1: RAW TITANIC DATA
========================================================= */
function Stage1RawData({ dataset, onNext }) {
  const sampleRows = dataset.slice(0, 8);
  const missingAgeCount = dataset.filter(r => !r.Age || isNaN(r.Age)).length;
  const missingEmbarkedCount = dataset.filter(r => !r.Embarked || r.Embarked.trim() === '').length;
  const missingCabinCount = dataset.filter(r => !r.Cabin || r.Cabin.trim() === '').length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
        <div style={{ background: '#0e182e', padding: '1.2rem', borderRadius: '10px', border: '1px solid #1f2e4d' }}>
          <div style={{ color: '#94a3b8', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Database size={16} /> Total Passengers
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#38bdf8', marginTop: '0.3rem' }}>
            {dataset.length}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Kaggle Titanic train.csv rows</div>
        </div>

        <div style={{ background: '#0e182e', padding: '1.2rem', borderRadius: '10px', border: '1px solid #1f2e4d' }}>
          <div style={{ color: '#fb7185', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <AlertTriangle size={16} /> Missing Age Values
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#fb7185', marginTop: '0.3rem' }}>
            {missingAgeCount} <span style={{ fontSize: '0.9rem', color: '#94a3b8' }}>({((missingAgeCount / dataset.length) * 100).toFixed(1)}%)</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Requires imputation</div>
        </div>

        <div style={{ background: '#0e182e', padding: '1.2rem', borderRadius: '10px', border: '1px solid #1f2e4d' }}>
          <div style={{ color: '#f59e0b', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <AlertTriangle size={16} /> Missing Embarked
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f59e0b', marginTop: '0.3rem' }}>
            {missingEmbarkedCount} <span style={{ fontSize: '0.9rem', color: '#94a3b8' }}>passengers</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Categorical port missing</div>
        </div>

        <div style={{ background: '#0e182e', padding: '1.2rem', borderRadius: '10px', border: '1px solid #1f2e4d' }}>
          <div style={{ color: '#a855f7', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Layers size={16} /> Missing Cabin
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#c084fc', marginTop: '0.3rem' }}>
            {missingCabinCount} <span style={{ fontSize: '0.9rem', color: '#94a3b8' }}>({((missingCabinCount / dataset.length) * 100).toFixed(1)}%)</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Heavy missingness</div>
        </div>
      </div>

      <div className="callout-box">
        <strong>Core ML Principle:</strong> Raw data contains useful information, but it is not always represented in the best form for machine learning algorithms. Strings, missing values, and varying numerical scales require systematic feature engineering.
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem' }}>
        <h3 style={{ fontSize: '1.1rem', color: '#e2e8f0', fontWeight: 600 }}>First 8 Passengers in Dataset</h3>
        <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Inspecting raw formats</span>
      </div>

      <div className="data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>PassengerId</th>
              <th>Survived</th>
              <th>Pclass</th>
              <th>Name</th>
              <th>Sex</th>
              <th>Age</th>
              <th>SibSp</th>
              <th>Parch</th>
              <th>Ticket</th>
              <th>Fare</th>
              <th>Cabin</th>
              <th>Embarked</th>
            </tr>
          </thead>
          <tbody>
            {sampleRows.map(row => (
              <tr key={row.PassengerId}>
                <td style={{ color: '#94a3b8' }}>{row.PassengerId}</td>
                <td>
                  <span style={{
                    padding: '0.15rem 0.5rem',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    background: row.Survived === '1' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                    color: row.Survived === '1' ? '#34d399' : '#f87171'
                  }}>
                    {row.Survived === '1' ? '1 (Yes)' : '0 (No)'}
                  </span>
                </td>
                <td>Class {row.Pclass}</td>
                <td style={{ maxWidth: '220px', overflow: 'hidden', textOverflow: 'ellipsis' }}>{row.Name}</td>
                <td>
                  <span style={{ color: row.Sex === 'female' ? '#f472b6' : '#60a5fa' }}>{row.Sex}</span>
                </td>
                <td>
                  {(!row.Age || row.Age === '') ? (
                    <span className="badge-nan">NaN ⚠</span>
                  ) : (
                    <span>{Number(row.Age).toFixed(0)}</span>
                  )}
                </td>
                <td>{row.SibSp}</td>
                <td>{row.Parch}</td>
                <td style={{ color: '#94a3b8' }}>{row.Ticket}</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>${Number(row.Fare).toFixed(2)}</td>
                <td style={{ color: '#64748b' }}>{row.Cabin || '—'}</td>
                <td>
                  {(!row.Embarked || row.Embarked.trim() === '') ? (
                    <span className="badge-nan">NaN ⚠</span>
                  ) : (
                    <span style={{ fontWeight: 600, color: '#38bdf8' }}>{row.Embarked}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1rem' }}>
        <button onClick={onNext} className="action-trigger-btn">
          Start Feature Engineering <ArrowRight size={18} />
        </button>
      </div>
    </div>
  );
}

/* =========================================================
   STAGE 2: MISSING VALUES
========================================================= */
function Stage2MissingValues({ dataset }) {
  const [isImputed, setIsImputed] = useState(false);

  const ageValues = dataset.map(d => d.Age).filter(v => v !== '' && !isNaN(v)).map(Number);
  const actualAgeMedian = calculateMedian(ageValues);

  const embarkedValues = dataset.map(d => d.Embarked).filter(v => v && v.trim() !== '');
  const actualEmbarkedMode = calculateMode(embarkedValues) || 'S';

  const missingAgeBefore = dataset.filter(r => !r.Age || isNaN(r.Age)).length;
  const missingEmbarkedBefore = dataset.filter(r => !r.Embarked || r.Embarked.trim() === '').length;

  const displayRows = dataset.filter(r => [1, 2, 5, 6, 17, 18].includes(Number(r.PassengerId)));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        <div style={{ background: '#0e182e', padding: '1.5rem', borderRadius: '12px', border: '1px solid #1f2e4d' }}>
          <h4 style={{ color: '#38bdf8', marginBottom: '1rem' }}>Age Column Imputation</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.9rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8' }}>
              <span>Missing Age Before:</span>
              <strong style={{ color: '#fb7185' }}>{missingAgeBefore} rows</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8' }}>
              <span>Calculated Median Age:</span>
              <strong style={{ color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>{actualAgeMedian} years</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8' }}>
              <span>Missing Age After:</span>
              <strong style={{ color: isImputed ? '#34d399' : '#94a3b8' }}>
                {isImputed ? '0 rows' : missingAgeBefore + ' rows'}
              </strong>
            </div>
          </div>
        </div>

        <div style={{ background: '#0e182e', padding: '1.5rem', borderRadius: '12px', border: '1px solid #1f2e4d' }}>
          <h4 style={{ color: '#38bdf8', marginBottom: '1rem' }}>Embarked Column Imputation</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.9rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8' }}>
              <span>Missing Embarked Before:</span>
              <strong style={{ color: '#fb7185' }}>{missingEmbarkedBefore} rows</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8' }}>
              <span>Most Frequent Port (Mode):</span>
              <strong style={{ color: '#38bdf8' }}>'{actualEmbarkedMode}' (Southampton)</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8' }}>
              <span>Missing Embarked After:</span>
              <strong style={{ color: isImputed ? '#34d399' : '#94a3b8' }}>
                {isImputed ? '0 rows' : missingEmbarkedBefore + ' rows'}
              </strong>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', margin: '0.5rem 0' }}>
        <button
          onClick={() => setIsImputed(!isImputed)}
          className="action-trigger-btn"
          style={{ background: isImputed ? '#10b981' : undefined }}
        >
          {isImputed ? <Check size={18} /> : <Sparkles size={18} />}
          {isImputed ? 'Imputed (Click to Toggle Before/After)' : 'Fill Missing Age & Embarked'}
        </button>
      </div>

      <div className="data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>PassengerId</th>
              <th>Name</th>
              <th>Sex</th>
              <th>Age (Target for Imputation)</th>
              <th>Embarked</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {displayRows.map(row => {
              const hasMissingAge = !row.Age || row.Age === '';
              return (
                <tr key={row.PassengerId}>
                  <td style={{ color: '#94a3b8' }}>#{row.PassengerId}</td>
                  <td>{row.Name}</td>
                  <td>{row.Sex}</td>
                  <td>
                    <AnimatePresence mode="wait">
                      {hasMissingAge ? (
                        isImputed ? (
                          <motion.span
                            key="filled"
                            initial={{ scale: 0.8, backgroundColor: 'rgba(56, 189, 248, 0.5)' }}
                            animate={{ scale: 1, backgroundColor: 'rgba(56, 189, 248, 0.15)' }}
                            className="badge-imputed"
                          >
                            {actualAgeMedian} (Median)
                          </motion.span>
                        ) : (
                          <motion.span key="nan" className="badge-nan">
                            NaN ⚠
                          </motion.span>
                        )
                      ) : (
                        <span>{row.Age}</span>
                      )}
                    </AnimatePresence>
                  </td>
                  <td>
                    {row.Embarked ? (
                      <span style={{ fontWeight: 600 }}>{row.Embarked}</span>
                    ) : (
                      isImputed ? (
                        <span className="badge-imputed">{actualEmbarkedMode} (Mode)</span>
                      ) : (
                        <span className="badge-nan">NaN ⚠</span>
                      )
                    )}
                  </td>
                  <td>
                    {hasMissingAge ? (
                      isImputed ? (
                        <span style={{ color: '#34d399', fontSize: '0.8rem' }}>✓ Imputed with Median</span>
                      ) : (
                        <span style={{ color: '#fb7185', fontSize: '0.8rem' }}>⚠ Needs Handling</span>
                      )
                    ) : (
                      <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Complete</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="callout-box">
        <strong>Evaluation Pipeline Rule (Preventing Data Leakage):</strong> Missing values must be handled before many machine learning algorithms can use the data effectively. In a proper model evaluation pipeline, imputation values should be learned only from the training split.
      </div>
    </div>
  );
}

/* =========================================================
   STAGE 3: FEATURE SCALING
========================================================= */
function Stage3Scaling({ dataset }) {
  const [selectedFeature, setSelectedFeature] = useState('Fare');
  const [method, setMethod] = useState('minmax');
  const [applied, setApplied] = useState(false);

  const rawValues = dataset
    .map(r => r[selectedFeature])
    .filter(v => v !== '' && !isNaN(v))
    .map(Number);

  const stats = calculateStats(rawValues);
  const samplePassengers = dataset.slice(0, 6);

  const chartData = samplePassengers.map(p => {
    const rawVal = Number(p[selectedFeature]) || 0;
    const scaledVal = method === 'minmax'
      ? minMaxScale(rawVal, stats.min, stats.max)
      : standardize(rawVal, stats.mean, stats.std);

    return {
      name: p.Name.split(',')[0],
      Original: Number(rawVal.toFixed(2)),
      Scaled: Number(scaledVal.toFixed(3))
    };
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', background: '#0e182e', padding: '1.25rem', borderRadius: '12px', border: '1px solid #1f2e4d', alignItems: 'center' }}>
        <div>
          <label style={{ fontSize: '0.85rem', color: '#94a3b8', display: 'block', marginBottom: '0.4rem' }}>Feature:</label>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {['Fare', 'Age'].map(feat => (
              <button
                key={feat}
                onClick={() => { setSelectedFeature(feat); setApplied(false); }}
                style={{
                  padding: '0.45rem 1rem',
                  borderRadius: '6px',
                  border: '1px solid',
                  borderColor: selectedFeature === feat ? '#38bdf8' : '#243552',
                  background: selectedFeature === feat ? 'rgba(56, 189, 248, 0.15)' : '#162238',
                  color: selectedFeature === feat ? '#38bdf8' : '#cbd5e1',
                  fontWeight: 600
                }}
              >
                {feat}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label style={{ fontSize: '0.85rem', color: '#94a3b8', display: 'block', marginBottom: '0.4rem' }}>Scaling Method:</label>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              onClick={() => { setMethod('minmax'); setApplied(false); }}
              style={{
                padding: '0.45rem 1rem',
                borderRadius: '6px',
                border: '1px solid',
                borderColor: method === 'minmax' ? '#818cf8' : '#243552',
                background: method === 'minmax' ? 'rgba(129, 140, 248, 0.15)' : '#162238',
                color: method === 'minmax' ? '#818cf8' : '#cbd5e1',
                fontWeight: 600
              }}
            >
              Min-Max Scaling
            </button>
            <button
              onClick={() => { setMethod('standard'); setApplied(false); }}
              style={{
                padding: '0.45rem 1rem',
                borderRadius: '6px',
                border: '1px solid',
                borderColor: method === 'standard' ? '#818cf8' : '#243552',
                background: method === 'standard' ? 'rgba(129, 140, 248, 0.15)' : '#162238',
                color: method === 'standard' ? '#818cf8' : '#cbd5e1',
                fontWeight: 600
              }}
            >
              Standardization
            </button>
          </div>
        </div>

        <div style={{ marginLeft: 'auto' }}>
          <button onClick={() => setApplied(true)} className="action-trigger-btn">
            <Play size={16} /> Apply Scaling
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1.5rem' }}>
        <div style={{ background: '#0e182e', padding: '1.25rem', borderRadius: '12px', border: '1px solid #1f2e4d' }}>
          <h4 style={{ color: '#38bdf8', marginBottom: '0.75rem' }}>Formula</h4>
          {method === 'minmax' ? (
            <div>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.92rem', color: '#818cf8', background: '#162238', padding: '0.75rem', borderRadius: '6px' }}>
                X_scaled = (X - X_min) / (X_max - X_min)
              </p>
              <div style={{ marginTop: '0.75rem', fontSize: '0.85rem', color: '#94a3b8' }}>
                <div>Min: <strong>${stats.min.toFixed(2)}</strong></div>
                <div>Max: <strong>${stats.max.toFixed(2)}</strong></div>
              </div>
            </div>
          ) : (
            <div>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.92rem', color: '#818cf8', background: '#162238', padding: '0.75rem', borderRadius: '6px' }}>
                Z = (X - μ) / σ
              </p>
              <div style={{ marginTop: '0.75rem', fontSize: '0.85rem', color: '#94a3b8' }}>
                <div>Mean (μ): <strong>{stats.mean.toFixed(2)}</strong></div>
                <div>Std Dev (σ): <strong>{stats.std.toFixed(2)}</strong></div>
              </div>
            </div>
          )}
        </div>

        <div className="data-table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Passenger</th>
                <th>ORIGINAL</th>
                <th></th>
                <th>SCALED</th>
              </tr>
            </thead>
            <tbody>
              {samplePassengers.map(p => {
                const rawVal = Number(p[selectedFeature]) || 0;
                const scaledVal = method === 'minmax'
                  ? minMaxScale(rawVal, stats.min, stats.max)
                  : standardize(rawVal, stats.mean, stats.std);

                return (
                  <tr key={p.PassengerId}>
                    <td>{p.Name.split(',')[0]}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{rawVal.toFixed(2)}</td>
                    <td style={{ color: '#38bdf8' }}>→</td>
                    <td>
                      <motion.span
                        key={`${p.PassengerId}-${applied}-${method}`}
                        initial={{ opacity: 0.3 }}
                        animate={{ opacity: 1 }}
                        style={{
                          fontFamily: 'var(--font-mono)',
                          fontWeight: 700,
                          color: applied ? '#34d399' : '#94a3b8'
                        }}
                      >
                        {applied ? scaledVal.toFixed(4) : '(Click Apply)'}
                      </motion.span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{ background: '#0e182e', padding: '1.25rem', borderRadius: '12px', border: '1px solid #1f2e4d' }}>
        <h4 style={{ color: '#e2e8f0', marginBottom: '1rem' }}>Before vs After Magnitudes</h4>
        <div style={{ height: '200px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <XAxis dataKey="name" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip contentStyle={{ background: '#162238', borderColor: '#243552', color: '#f1f5f9' }} />
              <Bar dataKey="Original" fill="#3b82f6" name={`Original ${selectedFeature}`} />
              {applied && <Bar dataKey="Scaled" fill="#10b981" name="Scaled" />}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="callout-box">
        <strong>Scaling Note:</strong> Scaling puts numerical features on comparable scales. It is especially useful for algorithms that are sensitive to feature magnitude, such as gradient descent optimizers or distance-based models. Note: Not all machine learning algorithms require feature scaling (for example, tree-based models like Random Forests are invariant to monotonic scaling).
      </div>
    </div>
  );
}

/* =========================================================
   STAGE 4: CATEGORICAL ENCODING
========================================================= */
function Stage4Encoding({ dataset }) {
  const [isSexEncoded, setIsSexEncoded] = useState(false);
  const [isEmbarkedSplit, setIsEmbarkedSplit] = useState(false);
  const samplePassengers = dataset.slice(0, 5);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <div style={{ background: '#0e182e', padding: '1.5rem', borderRadius: '12px', border: '1px solid #1f2e4d' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div>
            <h4 style={{ color: '#38bdf8', fontSize: '1.1rem' }}>1. Binary Encoding: Sex</h4>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Converts male/female string categories to 1/0 numerical values.</p>
          </div>
          <button onClick={() => setIsSexEncoded(!isSexEncoded)} className="action-trigger-btn">
            <Sparkles size={16} /> {isSexEncoded ? 'Reset' : 'Encode'}
          </button>
        </div>

        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
          {samplePassengers.map(p => {
            const encodedVal = p.Sex === 'male' ? 1 : 0;
            return (
              <motion.div
                key={p.PassengerId}
                layout
                style={{
                  background: '#162238',
                  border: '1px solid #243552',
                  padding: '0.75rem 1.25rem',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem'
                }}
              >
                <span style={{ color: '#94a3b8', fontSize: '0.85rem' }}>{p.Name.split(',')[0]}:</span>
                <motion.span
                  key={isSexEncoded ? 'num' : 'str'}
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                    color: isSexEncoded ? '#34d399' : (p.Sex === 'female' ? '#f472b6' : '#60a5fa')
                  }}
                >
                  {isSexEncoded ? `${encodedVal} (${p.Sex})` : p.Sex}
                </motion.span>
              </motion.div>
            );
          })}
        </div>
      </div>

      <div style={{ background: '#0e182e', padding: '1.5rem', borderRadius: '12px', border: '1px solid #1f2e4d' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div>
            <h4 style={{ color: '#818cf8', fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <GitFork size={20} /> 2. One-Hot Encoding: Embarked
            </h4>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Splitting single nominal column into binary indicator columns.</p>
          </div>
          <button
            onClick={() => setIsEmbarkedSplit(!isEmbarkedSplit)}
            className="action-trigger-btn"
            style={{ background: 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)' }}
          >
            <GitFork size={16} /> {isEmbarkedSplit ? 'Recombine Column' : 'Split Embarked Column'}
          </button>
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', margin: '1.5rem 0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
            <motion.div
              animate={{ scale: isEmbarkedSplit ? 0.95 : 1, borderColor: isEmbarkedSplit ? '#818cf8' : '#243552' }}
              style={{ background: '#162238', border: '2px solid #243552', padding: '1rem 1.75rem', borderRadius: '10px', textAlign: 'center' }}
            >
              <div style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Column</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#f1f5f9' }}>Embarked</div>
              <div style={{ fontSize: '0.8rem', color: '#38bdf8' }}>['S', 'C', 'Q']</div>
            </motion.div>

            <motion.div
              animate={{ x: isEmbarkedSplit ? 10 : 0, opacity: isEmbarkedSplit ? 1 : 0.4 }}
              style={{ fontSize: '2rem', color: '#6366f1' }}
            >
              ───────►
            </motion.div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {['Embarked_C', 'Embarked_Q', 'Embarked_S'].map((col, idx) => (
                <motion.div
                  key={col}
                  initial={{ opacity: 0.3, x: -15 }}
                  animate={{ opacity: isEmbarkedSplit ? 1 : 0.3, x: isEmbarkedSplit ? 0 : -15 }}
                  transition={{ delay: idx * 0.1 }}
                  style={{
                    background: isEmbarkedSplit ? 'rgba(99, 102, 241, 0.2)' : '#162238',
                    border: '1px solid',
                    borderColor: isEmbarkedSplit ? '#818cf8' : '#243552',
                    padding: '0.45rem 1.25rem',
                    borderRadius: '8px',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.9rem',
                    color: isEmbarkedSplit ? '#c7d2fe' : '#64748b'
                  }}
                >
                  ✨ {col}
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        <div className="data-table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Passenger</th>
                <th>Raw Embarked</th>
                {isEmbarkedSplit ? (
                  <>
                    <th style={{ color: '#818cf8' }}>Embarked_C</th>
                    <th style={{ color: '#818cf8' }}>Embarked_Q</th>
                    <th style={{ color: '#818cf8' }}>Embarked_S</th>
                  </>
                ) : (
                  <th>One-Hot State</th>
                )}
              </tr>
            </thead>
            <tbody>
              {samplePassengers.map(p => {
                const emb = p.Embarked || 'S';
                const isC = emb === 'C' ? 1 : 0;
                const isQ = emb === 'Q' ? 1 : 0;
                const isS = emb === 'S' ? 1 : 0;

                return (
                  <tr key={p.PassengerId}>
                    <td>{p.Name.split(',')[0]}</td>
                    <td><span style={{ fontWeight: 700, color: '#38bdf8' }}>{emb}</span></td>
                    {isEmbarkedSplit ? (
                      <>
                        <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: isC ? '#34d399' : '#64748b' }}>{isC}</td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: isQ ? '#34d399' : '#64748b' }}>{isQ}</td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: isS ? '#34d399' : '#64748b' }}>{isS}</td>
                      </>
                    ) : (
                      <td style={{ color: '#64748b' }}>Raw column</td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="callout-box">
        <strong>Categorical Encoding Principle:</strong> One-hot encoding represents categories numerically without creating a false order between them.
      </div>
    </div>
  );
}

/* =========================================================
   STAGE 5: CREATE FamilySize
========================================================= */
function Stage5FamilySize({ dataset }) {
  const [created, setCreated] = useState(false);
  const exemplar = dataset.find(r => Number(r.PassengerId) === 8) || dataset[7];

  const sibsp = Number(exemplar.SibSp);
  const parch = Number(exemplar.Parch);
  const familySize = sibsp + parch + 1;
  const sampleRows = dataset.slice(7, 13);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <div style={{ background: '#0e182e', padding: '2rem', borderRadius: '16px', border: '1px solid #1f2e4d', textAlign: 'center' }}>
        <h3 style={{ fontSize: '1.4rem', color: '#ffffff', marginBottom: '1.25rem' }}>
          Combining Siblings/Spouses (SibSp) and Parents/Children (Parch)
        </h3>

        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '1.25rem', flexWrap: 'wrap' }}>
          <div style={{ background: '#162238', border: '1px solid #243552', padding: '1rem 1.5rem', borderRadius: '10px' }}>
            <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>SibSp</div>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: '#38bdf8' }}>{sibsp}</div>
          </div>

          <Plus size={24} style={{ color: '#818cf8' }} />

          <div style={{ background: '#162238', border: '1px solid #243552', padding: '1rem 1.5rem', borderRadius: '10px' }}>
            <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Parch</div>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: '#38bdf8' }}>{parch}</div>
          </div>

          <Plus size={24} style={{ color: '#818cf8' }} />

          <div style={{ background: '#162238', border: '1px solid #243552', padding: '1rem 1.5rem', borderRadius: '10px' }}>
            <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>+1 (Self)</div>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: '#10b981' }}>1</div>
          </div>

          <div style={{ fontSize: '1.8rem', color: '#818cf8' }}>=</div>

          <motion.div
            animate={{
              scale: created ? [1, 1.15, 1] : 1,
              boxShadow: created ? '0 0 25px rgba(16, 185, 129, 0.4)' : 'none'
            }}
            transition={{ duration: 0.5 }}
            style={{
              background: created ? 'rgba(16, 185, 129, 0.15)' : '#162238',
              border: '2px solid',
              borderColor: created ? '#10b981' : '#243552',
              padding: '1rem 1.75rem',
              borderRadius: '10px'
            }}
          >
            <div style={{ fontSize: '0.8rem', color: created ? '#34d399' : '#94a3b8' }}>FamilySize</div>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: created ? '#34d399' : '#64748b' }}>
              {created ? familySize : '?'}
            </div>
          </motion.div>
        </div>

        <div style={{ marginTop: '1.5rem' }}>
          <button onClick={() => setCreated(!created)} className="action-trigger-btn">
            <Sparkles size={18} /> {created ? 'Reset' : 'Calculate & Slide In FamilySize'}
          </button>
        </div>
      </div>

      <div className="data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Passenger</th>
              <th>SibSp</th>
              <th>Parch</th>
              <th>Self</th>
              <AnimatePresence>
                {created && (
                  <motion.th
                    initial={{ opacity: 0, width: 0 }}
                    animate={{ opacity: 1, width: 'auto' }}
                    exit={{ opacity: 0, width: 0 }}
                    style={{ background: '#064e3b', color: '#34d399' }}
                  >
                    ✨ FamilySize — NEW FEATURE
                  </motion.th>
                )}
              </AnimatePresence>
            </tr>
          </thead>
          <tbody>
            {sampleRows.map(p => {
              const s = Number(p.SibSp) || 0;
              const pa = Number(p.Parch) || 0;
              const fs = s + pa + 1;

              return (
                <tr key={p.PassengerId}>
                  <td>{p.Name}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{s}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{pa}</td>
                  <td style={{ color: '#10b981', fontFamily: 'var(--font-mono)' }}>+1</td>
                  <AnimatePresence>
                    {created && (
                      <motion.td
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0 }}
                        style={{
                          background: 'rgba(16, 185, 129, 0.08)',
                          fontWeight: 700,
                          color: '#34d399',
                          fontFamily: 'var(--font-mono)'
                        }}
                      >
                        {fs}
                      </motion.td>
                    )}
                  </AnimatePresence>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="callout-box strong">
        <strong>Core Concept:</strong> Feature engineering can create new information by combining existing variables. The +1 represents the passenger themself.
      </div>
    </div>
  );
}

/* =========================================================
   STAGE 6: CREATE IsAlone
========================================================= */
function Stage6IsAlone({ dataset }) {
  const [created, setCreated] = useState(false);
  const samplePassengers = dataset.slice(0, 6);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <div style={{ background: '#0e182e', padding: '1.75rem', borderRadius: '14px', border: '1px solid #1f2e4d' }}>
        <h3 style={{ color: '#38bdf8', fontSize: '1.2rem', marginBottom: '1.25rem', textAlign: 'center' }}>
          Deriving Travelling Solo Indicator
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          <div style={{ background: '#162238', border: '1px solid #243552', padding: '1.25rem', borderRadius: '10px', textAlign: 'center' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.1rem' }}>FamilySize = 1</div>
            <div style={{ color: '#818cf8', margin: '0.4rem 0' }}>↓</div>
            <div style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Travelling alone?</div>
            <div style={{ color: '#10b981', fontWeight: 700 }}>YES</div>
            <div style={{ color: '#818cf8', margin: '0.4rem 0' }}>↓</div>
            <div style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#34d399', padding: '0.4rem', borderRadius: '6px', fontWeight: 800 }}>
              IsAlone = 1
            </div>
          </div>

          <div style={{ background: '#162238', border: '1px solid #243552', padding: '1.25rem', borderRadius: '10px', textAlign: 'center' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.1rem' }}>FamilySize = 4</div>
            <div style={{ color: '#818cf8', margin: '0.4rem 0' }}>↓</div>
            <div style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Travelling alone?</div>
            <div style={{ color: '#fb7185', fontWeight: 700 }}>NO</div>
            <div style={{ color: '#818cf8', margin: '0.4rem 0' }}>↓</div>
            <div style={{ background: 'rgba(244, 63, 94, 0.2)', color: '#fb7185', padding: '0.4rem', borderRadius: '6px', fontWeight: 800 }}>
              IsAlone = 0
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '1.5rem' }}>
          <button onClick={() => setCreated(!created)} className="action-trigger-btn">
            <Sparkles size={18} /> {created ? 'Reset' : 'Generate ✨ IsAlone Feature'}
          </button>
        </div>
      </div>

      <div className="callout-box strong" style={{ fontSize: '1.05rem', lineHeight: '1.6' }}>
        <strong>Important Explanation:</strong> The original dataset did not explicitly contain an “IsAlone” column. We derived this information from existing features. This is a clear example of feature creation.
      </div>

      <div className="data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Passenger</th>
              <th>SibSp</th>
              <th>Parch</th>
              <th>FamilySize</th>
              <th>Travelling Alone?</th>
              {created && <th style={{ color: '#34d399' }}>✨ IsAlone — NEW FEATURE</th>}
            </tr>
          </thead>
          <tbody>
            {samplePassengers.map(p => {
              const sibsp = Number(p.SibSp) || 0;
              const parch = Number(p.Parch) || 0;
              const famSize = sibsp + parch + 1;
              const isAlone = famSize === 1 ? 1 : 0;

              return (
                <tr key={p.PassengerId}>
                  <td>{p.Name}</td>
                  <td>{sibsp}</td>
                  <td>{parch}</td>
                  <td style={{ fontWeight: 600 }}>{famSize}</td>
                  <td style={{ color: isAlone ? '#38bdf8' : '#f59e0b' }}>
                    {isAlone ? 'YES' : 'NO'}
                  </td>
                  {created && (
                    <motion.td
                      initial={{ scale: 0.8, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontWeight: 700,
                        color: isAlone ? '#34d399' : '#64748b',
                        background: isAlone ? 'rgba(16, 185, 129, 0.1)' : 'transparent'
                      }}
                    >
                      {isAlone}
                    </motion.td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* =========================================================
   STAGE 7: EXTRACT TITLE FROM NAME
========================================================= */
function Stage7TitleExtract({ dataset }) {
  const [extracted, setExtracted] = useState(false);

  const titleCounts = {};
  dataset.forEach(r => {
    const title = extractTitle(r.Name);
    titleCounts[title] = (titleCounts[title] || 0) + 1;
  });

  const chartData = Object.keys(titleCounts).map(t => ({
    title: t,
    passengers: titleCounts[t]
  })).sort((a, b) => b.passengers - a.passengers);

  const targetIds = [1, 2, 3, 8];
  const displayPassengers = dataset.filter(r => targetIds.includes(Number(r.PassengerId)));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <div style={{ background: '#0e182e', padding: '1.75rem', borderRadius: '14px', border: '1px solid #1f2e4d' }}>
        <h3 style={{ color: '#38bdf8', fontSize: '1.2rem', marginBottom: '0.5rem' }}>
          Extracting Title from Passenger Name Text
        </h3>

        <div style={{ background: '#162238', padding: '1.5rem', borderRadius: '10px', textAlign: 'center' }}>
          <div style={{ fontSize: '1.25rem', fontFamily: 'var(--font-mono)' }}>
            <span style={{ color: extracted ? '#475569' : '#f1f5f9', transition: 'color 0.4s' }}>Braund, </span>
            <motion.span
              animate={{ color: extracted ? '#38bdf8' : '#f1f5f9', scale: extracted ? 1.25 : 1 }}
              style={{ fontWeight: 800, display: 'inline-block' }}
            >
              Mr
            </motion.span>
            <span style={{ color: extracted ? '#475569' : '#f1f5f9', transition: 'color 0.4s' }}>. Owen Harris</span>
          </div>

          <div style={{ margin: '0.75rem 0', color: '#818cf8' }}>↓</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: extracted ? '#38bdf8' : '#64748b' }}>Mr</div>
          <div style={{ margin: '0.75rem 0', color: '#818cf8' }}>↓</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: extracted ? '#34d399' : '#64748b' }}>
            {extracted ? 'Title = Mr' : 'Title'}
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '1.25rem' }}>
          <button onClick={() => setExtracted(!extracted)} className="action-trigger-btn">
            <Sparkles size={16} /> {extracted ? 'Reset' : 'Extract Title Programmatically'}
          </button>
        </div>
      </div>

      <div className="data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Passenger Name</th>
              <th>Extracted Title</th>
            </tr>
          </thead>
          <tbody>
            {displayPassengers.map(p => {
              const title = extractTitle(p.Name);
              return (
                <tr key={p.PassengerId}>
                  <td>{p.Name}</td>
                  <td>
                    {extracted ? (
                      <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="badge-new">
                        {title}
                      </motion.span>
                    ) : (
                      <span style={{ color: '#64748b' }}>Unextracted</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div style={{ background: '#0e182e', padding: '1.25rem', borderRadius: '12px', border: '1px solid #1f2e4d' }}>
        <h4 style={{ color: '#e2e8f0', marginBottom: '0.75rem' }}>Common Passenger Titles Distribution</h4>
        <div style={{ height: '180px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <XAxis dataKey="title" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip contentStyle={{ background: '#162238', borderColor: '#243552', color: '#f1f5f9' }} />
              <Bar dataKey="passengers" fill="#818cf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="callout-box">
        <strong>Text Feature Extraction:</strong> Useful structured information can sometimes be extracted from raw text features. Titles convey both gender and age dynamics (e.g., 'Master' indicates male children).
      </div>
    </div>
  );
}

/* =========================================================
   STAGE 8: LOG TRANSFORMATION OF FARE
========================================================= */
function Stage8LogFare({ dataset }) {
  const [transformed, setTransformed] = useState(false);

  const rawFares = dataset.map(r => r.Fare).filter(f => f !== '' && !isNaN(f)).map(Number);
  const logFares = rawFares.map(f => Math.log(1 + Math.max(0, f)));

  const skewnessBefore = calculateSkewness(rawFares);
  const skewnessAfter = calculateSkewness(logFares);

  const rawHistogram = buildHistogram(rawFares, 12, false);
  const logHistogram = buildHistogram(rawFares, 12, true);
  const activeHistogram = transformed ? logHistogram : rawHistogram;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1.25rem' }}>
        <div style={{ background: '#0e182e', padding: '1.25rem', borderRadius: '12px', border: '1px solid #1f2e4d' }}>
          <div style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Skewness Before:</div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#fb7185', marginTop: '0.25rem' }}>
            +{skewnessBefore.toFixed(2)}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Original Fare</div>
        </div>

        <div style={{ background: '#0e182e', padding: '1.25rem', borderRadius: '12px', border: '1px solid #1f2e4d' }}>
          <div style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Skewness After log(1 + Fare):</div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#34d399', marginTop: '0.25rem' }}>
            +{skewnessAfter.toFixed(2)}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Reduced positive skewness</div>
        </div>

        <div style={{ background: '#0e182e', padding: '1.25rem', borderRadius: '12px', border: '1px solid #1f2e4d', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <button onClick={() => setTransformed(!transformed)} className="action-trigger-btn">
            <Play size={16} /> {transformed ? 'Show Original' : 'Apply Log Transformation'}
          </button>
        </div>
      </div>

      <div style={{ background: '#0e182e', padding: '1.5rem', borderRadius: '14px', border: '1px solid #1f2e4d' }}>
        <h4 style={{ color: '#ffffff', fontSize: '1.1rem', marginBottom: '1rem' }}>
          {transformed ? 'log(1 + Fare) Distribution' : 'Original Fare Distribution'}
        </h4>
        <div style={{ height: '240px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={activeHistogram}>
              <XAxis dataKey="binLabel" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis stroke="#64748b" />
              <Tooltip contentStyle={{ background: '#162238', borderColor: '#243552', color: '#f1f5f9' }} />
              <Bar dataKey="count" fill={transformed ? '#10b981' : '#f43f5e'} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="callout-box">
        <strong>Skewness & Transformation:</strong> Fare is positively skewed because some passengers paid much higher fares. Log transformation compresses large values and can reduce positive skewness. (Note: Log transformation does not guarantee that the transformed data will become perfectly normally distributed).
      </div>
    </div>
  );
}

/* =========================================================
   STAGE 9: FEATURE SELECTION
========================================================= */
const INITIAL_FEATURES = [
  { name: 'Pclass', type: 'Kept ✓', status: 'keep' },
  { name: 'Age', type: 'Kept ✓', status: 'keep' },
  { name: 'Sex encoded', type: 'Kept ✓', status: 'keep' },
  { name: 'Embarked encoded', type: 'Kept ✓', status: 'keep' },
  { name: 'Fare_Log', type: 'Kept ✓', status: 'keep' },
  { name: 'FamilySize', type: 'Engineered ✨', status: 'keep' },
  { name: 'IsAlone', type: 'Engineered ✨', status: 'keep' },
  { name: 'Title', type: 'Engineered ✨', status: 'keep' },
  { name: 'PassengerId', type: 'Removed ?', status: 'drop', reason: 'Arbitrary identifier' },
  { name: 'Ticket', type: 'Removed ?', status: 'drop', reason: 'High-cardinality noise' },
  { name: 'Raw Name', type: 'Removed ?', status: 'drop', reason: 'Replaced by Title' }
];

function Stage9Selection() {
  const [selectedOnly, setSelectedOnly] = useState(false);
  const displayed = selectedOnly ? INITIAL_FEATURES.filter(f => f.status === 'keep') : INITIAL_FEATURES;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        <div style={{ background: '#0e182e', padding: '1.25rem', borderRadius: '12px', border: '1px solid #1f2e4d' }}>
          <h4 style={{ color: '#38bdf8', marginBottom: '0.4rem' }}>Feature Engineering</h4>
          <p style={{ color: '#cbd5e1', fontSize: '0.88rem' }}>Creates or transforms information.</p>
        </div>
        <div style={{ background: '#0e182e', padding: '1.25rem', borderRadius: '12px', border: '1px solid #1f2e4d' }}>
          <h4 style={{ color: '#818cf8', marginBottom: '0.4rem' }}>Feature Selection</h4>
          <p style={{ color: '#cbd5e1', fontSize: '0.88rem' }}>Chooses which available features are retained.</p>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <button onClick={() => setSelectedOnly(!selectedOnly)} className="action-trigger-btn">
          <Filter size={16} /> {selectedOnly ? 'Show All Available Features' : 'Choose Final Features'}
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '1rem' }}>
        <AnimatePresence>
          {displayed.map(feat => {
            const isKeep = feat.status === 'keep';
            return (
              <motion.div
                key={feat.name}
                layout
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                style={{
                  background: isKeep ? '#162238' : 'rgba(30, 41, 59, 0.4)',
                  border: '1px solid',
                  borderColor: isKeep ? '#243552' : '#334155',
                  padding: '1.2rem',
                  borderRadius: '10px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontWeight: 700, color: isKeep ? '#f1f5f9' : '#94a3b8' }}>{feat.name}</span>
                  <span style={{ color: isKeep ? '#34d399' : '#fb7185', fontWeight: 700 }}>{feat.type}</span>
                </div>
                {feat.reason && <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '0.5rem' }}>{feat.reason}</p>}
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      <div className="callout-box">
        <strong>Representation Choice:</strong> These features are excluded from our chosen representation. This does not mean they can never contain useful information; other feature-engineering approaches could extract additional information from them.
      </div>
    </div>
  );
}

/* =========================================================
   STAGE 10: BEFORE VS AFTER
========================================================= */
function Stage10BeforeAfter() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '1.5rem', alignItems: 'center' }}>
        <div style={{ background: 'rgba(239, 68, 68, 0.04)', border: '1px solid rgba(239, 68, 68, 0.25)', padding: '1.75rem', borderRadius: '16px' }}>
          <h3 style={{ color: '#f87171', fontSize: '1.25rem', marginBottom: '1rem' }}>BEFORE: Raw Titanic Dataset</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
            {[
              { col: 'PassengerId', tag: 'Identifier' },
              { col: 'Pclass', tag: 'Raw class' },
              { col: 'Name', tag: '💡 Information hidden inside Name' },
              { col: 'Sex', tag: '🔤 Categories' },
              { col: 'Age', tag: '⚠ Missing values' },
              { col: 'SibSp', tag: 'Uncombined count' },
              { col: 'Parch', tag: 'Uncombined count' },
              { col: 'Ticket', tag: 'Alphanumeric string' },
              { col: 'Fare', tag: '📈 Skewed Fare' },
              { col: 'Cabin', tag: 'Sparse data' },
              { col: 'Embarked', tag: '🔤 Categories' }
            ].map(item => (
              <div key={item.col} style={{ background: '#0e182e', padding: '0.55rem 0.85rem', borderRadius: '6px', display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                <span style={{ fontFamily: 'var(--font-mono)' }}>{item.col}</span>
                <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>{item.tag}</span>
              </div>
            ))}
          </div>
        </div>

        <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
          <motion.div
            animate={{ scale: [1, 1.08, 1] }}
            transition={{ repeat: Infinity, duration: 2 }}
            style={{
              background: 'linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%)',
              color: 'white',
              padding: '1rem',
              borderRadius: '50%',
              boxShadow: '0 0 20px rgba(14, 165, 233, 0.4)'
            }}
          >
            <Sparkles size={24} />
          </motion.div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: '#38bdf8', fontWeight: 700 }}>
            RAW DATA<br />↓<br />⚙️ FEATURE ENGINEERING<br />↓<br />ML-READY DATA
          </div>
        </div>

        <div style={{ background: 'rgba(16, 185, 129, 0.04)', border: '1px solid rgba(16, 185, 129, 0.25)', padding: '1.75rem', borderRadius: '16px' }}>
          <h3 style={{ color: '#34d399', fontSize: '1.25rem', marginBottom: '1rem' }}>AFTER: ML-Ready Features</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
            {[
              { col: 'Pclass', tag: 'Normalized' },
              { col: 'Age_scaled', tag: 'Imputed & Scaled' },
              { col: 'Fare_log', tag: 'Skewness reduced' },
              { col: 'Sex_encoded', tag: 'Binary 0/1' },
              { col: 'Embarked_C', tag: 'One-Hot 1/0' },
              { col: 'Embarked_Q', tag: 'One-Hot 1/0' },
              { col: 'Embarked_S', tag: 'One-Hot 1/0' },
              { col: 'FamilySize ✨', tag: 'Created (SibSp+Parch+1)', glow: true },
              { col: 'IsAlone ✨', tag: 'Created (Binary flag)', glow: true },
              { col: 'Title ✨', tag: 'Created (Extracted from Name)', glow: true }
            ].map(item => (
              <motion.div
                key={item.col}
                animate={item.glow ? {
                  boxShadow: ['0 0 0px rgba(52, 211, 153, 0)', '0 0 12px rgba(52, 211, 153, 0.3)', '0 0 0px rgba(52, 211, 153, 0)']
                } : {}}
                transition={{ repeat: Infinity, duration: 2.5 }}
                style={{
                  background: item.glow ? 'rgba(16, 185, 129, 0.12)' : '#0e182e',
                  border: item.glow ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid #1f2e4d',
                  padding: '0.55rem 0.85rem',
                  borderRadius: '6px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '0.85rem'
                }}
              >
                <span style={{ fontFamily: 'var(--font-mono)', color: item.glow ? '#34d399' : '#f1f5f9', fontWeight: 600 }}>
                  {item.col}
                </span>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{item.tag}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ textAlign: 'center', marginTop: '1rem' }}>
        <h1 style={{ fontSize: '1.9rem', fontWeight: 800 }}>Raw Data → Meaningful Features → ML-Ready Data</h1>
        <p style={{ color: '#cbd5e1', marginTop: '0.5rem' }}>
          Feature Engineering improves the representation of information supplied to a machine learning model.
        </p>
      </div>
    </div>
  );
}

/* =========================================================
   STAGE 11: DOES FEATURE ENGINEERING HELP?
========================================================= */
function Stage11MLComparison({ dataset }) {
  const [results, setResults] = useState(null);

  useEffect(() => {
    if (dataset && dataset.length > 0) {
      const res = runLogisticRegressionExperiment(dataset);
      setResults(res);
    }
  }, [dataset]);

  if (!results) {
    return <div style={{ color: '#94a3b8' }}>Evaluating Logistic Regression models from train.csv...</div>;
  }

  const { modelA, modelB, trainSize, testSize } = results;
  const isEngineeredBetter = modelB.accuracy >= modelA.accuracy;

  const metricsTable = [
    { label: 'Accuracy', basic: (modelA.accuracy * 100).toFixed(1) + '%', eng: (modelB.accuracy * 100).toFixed(1) + '%' },
    { label: 'Precision', basic: (modelA.precision * 100).toFixed(1) + '%', eng: (modelB.precision * 100).toFixed(1) + '%' },
    { label: 'Recall', basic: (modelA.recall * 100).toFixed(1) + '%', eng: (modelB.recall * 100).toFixed(1) + '%' },
    { label: 'F1 Score', basic: (modelA.f1 * 100).toFixed(1) + '%', eng: (modelB.f1 * 100).toFixed(1) + '%' }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#0e182e', padding: '1.25rem', borderRadius: '12px', border: '1px solid #1f2e4d' }}>
        <div>
          <h4 style={{ color: '#38bdf8', fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={18} /> Model: Logistic Regression
          </h4>
          <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
            Target: <code>Survived</code> (never included as an input feature).
          </p>
        </div>
        <div style={{ textAlign: 'right', fontSize: '0.85rem', color: '#cbd5e1' }}>
          <div>Train split: {trainSize} | Test split: {testSize}</div>
        </div>
      </div>

      <div className="data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th style={{ color: '#94a3b8' }}>Model A — Basic Features</th>
              <th style={{ color: '#34d399' }}>Model B — Engineered Features</th>
            </tr>
          </thead>
          <tbody>
            {metricsTable.map(m => (
              <tr key={m.label}>
                <td style={{ fontWeight: 600 }}>{m.label}</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>{m.basic}</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#34d399' }}>{m.eng}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="callout-box" style={{ background: 'rgba(16, 185, 129, 0.08)', borderLeftColor: '#10b981' }}>
        <strong style={{ color: '#34d399' }}>Validation Finding:</strong>
        <p style={{ color: '#cbd5e1', marginTop: '0.25rem' }}>
          {isEngineeredBetter
            ? 'For this experiment, the engineered representation improved validation performance.'
            : 'Feature engineering does not guarantee better performance. Features must be evaluated experimentally.'}
        </p>
      </div>

      <div style={{ background: '#0e182e', padding: '1rem', borderRadius: '10px', border: '1px solid #1f2e4d', display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
        <ShieldCheck size={20} style={{ color: '#38bdf8' }} />
        <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
          Imputation/scaling/encoding parameters used for ML evaluation were learned from the training data only.
        </span>
      </div>

      {/* FINAL SUMMARY 3 CARDS */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.25rem', marginTop: '1rem' }}>
        <div style={{ background: '#0e182e', border: '1px solid #1f2e4d', borderRadius: '12px', padding: '1.5rem' }}>
          <div style={{ color: '#38bdf8', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <RefreshCw size={20} />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800 }}>🔄 TRANSFORM</h3>
          </div>
          <p style={{ color: '#cbd5e1', fontSize: '0.85rem' }}>Change how an existing feature is represented.</p>
          <div style={{ marginTop: '0.75rem', fontSize: '0.8rem', color: '#94a3b8' }}>
            <strong>Examples:</strong> Scaling and Fare log transformation.
          </div>
        </div>

        <div style={{ background: '#0e182e', border: '1px solid #1f2e4d', borderRadius: '12px', padding: '1.5rem' }}>
          <div style={{ color: '#10b981', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Sparkles size={20} />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800 }}>✨ CREATE</h3>
          </div>
          <p style={{ color: '#cbd5e1', fontSize: '0.85rem' }}>Derive new information from existing features.</p>
          <div style={{ marginTop: '0.75rem', fontSize: '0.8rem', color: '#94a3b8' }}>
            <strong>Examples:</strong> FamilySize, IsAlone and Title.
          </div>
        </div>

        <div style={{ background: '#0e182e', border: '1px solid #1f2e4d', borderRadius: '12px', padding: '1.5rem' }}>
          <div style={{ color: '#a855f7', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Scissors size={20} />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800 }}>✂️ SELECT</h3>
          </div>
          <p style={{ color: '#cbd5e1', fontSize: '0.85rem' }}>Choose which features are supplied to the model.</p>
          <div style={{ marginTop: '0.75rem', fontSize: '0.8rem', color: '#94a3b8' }}>
            <strong>Examples:</strong> Dropping PassengerId and Ticket.
          </div>
        </div>
      </div>

      <div style={{ textAlign: 'center', marginTop: '1rem' }}>
        <h1 style={{ fontSize: '2.2rem', fontWeight: 900 }}>FEATURE ENGINEERING</h1>
        <h3 style={{ color: '#38bdf8' }}>Raw Data → Meaningful Features → ML-Ready Data</h3>
      </div>
    </div>
  );
}

/* =========================================================
   WRAPPER COMPONENTS
========================================================= */
const STAGES = [
  { id: 1, label: 'Raw Data' },
  { id: 2, label: 'Missing Values' },
  { id: 3, label: 'Scaling' },
  { id: 4, label: 'Encoding' },
  { id: 5, label: 'Create Features' },
  { id: 6, label: 'IsAlone' },
  { id: 7, label: 'Title' },
  { id: 8, label: 'Transform' },
  { id: 9, label: 'Select' },
  { id: 10, label: 'Before vs After' },
  { id: 11, label: 'ML Ready' }
];

function PipelineNav({ currentStage, onSelectStage }) {
  return (
    <nav className="pipeline-bar">
      {STAGES.map((s, idx) => {
        const isActive = s.id === currentStage;
        const isCompleted = s.id < currentStage;
        return (
          <React.Fragment key={s.id}>
            <button
              onClick={() => onSelectStage(s.id)}
              className={`pipeline-node ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
            >
              <span>{s.label}</span>
            </button>
            {idx < STAGES.length - 1 && (
              <ChevronRight className="pipeline-arrow" size={14} />
            )}
          </React.Fragment>
        );
      })}
    </nav>
  );
}

function StageCardWrapper({ stageNumber, title, whatText, whyText, resultText, children }) {
  return (
    <motion.section
      key={stageNumber}
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.3 }}
      className="stage-main-card"
    >
      <div>
        <span className="stage-badge">Stage {stageNumber} of 11</span>
        <h2 className="stage-title">{title}</h2>
      </div>

      <div className="trio-container">
        <div className="trio-box what">
          <div className="trio-label"><HelpCircle size={14} /> WHAT?</div>
          <p className="trio-text">{whatText}</p>
        </div>
        <div className="trio-box why">
          <div className="trio-label"><Sparkles size={14} /> WHY?</div>
          <p className="trio-text">{whyText}</p>
        </div>
        <div className="trio-box result">
          <div className="trio-label"><CheckCircle2 size={14} /> RESULT</div>
          <p className="trio-text">{resultText}</p>
        </div>
      </div>

      <div>{children}</div>
    </motion.section>
  );
}

/* =========================================================
   MAIN APP CONTROLLER
========================================================= */
export default function App() {
  const [currentStage, setCurrentStage] = useState(1);
  const [dataset, setDataset] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [presentationMode, setPresentationMode] = useState(false);

  useEffect(() => {
    loadTitanicDataset()
      .then(data => {
        setDataset(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const handleNext = () => {
    if (currentStage < 11) setCurrentStage(prev => prev + 1);
  };

  const handlePrev = () => {
    if (currentStage > 1) setCurrentStage(prev => prev - 1);
  };

  const handleReset = () => {
    setCurrentStage(1);
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', background: '#0a0f1d' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: '#38bdf8', fontSize: '1.5rem', fontWeight: 700 }}>Loading Titanic Dataset...</div>
          <div style={{ color: '#94a3b8', marginTop: '0.5rem' }}>Reading /public/train.csv</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', background: '#0a0f1d' }}>
        <div style={{ background: '#1e293b', padding: '2rem', borderRadius: '12px', border: '1px solid #ef4444', maxWidth: '500px' }}>
          <h2 style={{ color: '#ef4444', marginBottom: '1rem' }}>Failed to Load train.csv</h2>
          <p style={{ color: '#cbd5e1', fontSize: '0.9rem' }}>
            Please make sure that <code>train.csv</code> is placed in your project's <code>public/train.csv</code> directory.
          </p>
          <div style={{ color: '#94a3b8', fontSize: '0.8rem', marginTop: '1rem' }}>Error details: {error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`app-container ${presentationMode ? 'presentation-mode' : ''}`}>
      <header className="app-header">
        <div className="header-title-group">
          <h1>Feature Engineering Visualizer</h1>
          <p>Transforming Raw Titanic Data into ML-Ready Features</p>
        </div>

        <button
          onClick={() => setPresentationMode(!presentationMode)}
          className={`btn-presentation ${presentationMode ? 'active' : ''}`}
        >
          <Mic size={16} /> 🎤 Presentation Mode
        </button>
      </header>

      {/* TOP PIPELINE */}
      <PipelineNav currentStage={currentStage} onSelectStage={(id) => setCurrentStage(id)} />

      {/* ACTIVE STAGE COMPONENT */}
      <main style={{ marginBottom: '4rem' }}>
        <AnimatePresence mode="wait">
          {currentStage === 1 && (
            <StageCardWrapper
              stageNumber={1}
              title="Why Do We Need Feature Engineering?"
              whatText="Inspecting the raw, unprocessed Titanic training dataset in its native schema."
              whyText="Raw features have missing values, unencoded categories, and inconsistent scales."
              resultText="Identified critical missingness in Age and Embarked, plus categorical columns needing encoding."
            >
              <Stage1RawData dataset={dataset} onNext={handleNext} />
            </StageCardWrapper>
          )}

          {currentStage === 2 && (
            <StageCardWrapper
              stageNumber={2}
              title="Handle Missing Values"
              whatText="Focus on Age and Embarked. Imputing missing values using the median and mode."
              whyText="Missing values must be handled before many machine learning algorithms can use the data effectively."
              resultText="Missing Age rows reduced to 0; missing Embarked values filled with the most frequent category."
            >
              <Stage2MissingValues dataset={dataset} />
            </StageCardWrapper>
          )}

          {currentStage === 3 && (
            <StageCardWrapper
              stageNumber={3}
              title="Feature Scaling"
              whatText="Allowing selection between Min-Max Scaling and Standardization for Age and Fare."
              whyText="Scaling puts numerical features on comparable scales. It is especially useful for algorithms that are sensitive to feature magnitude."
              resultText="Original values scaled into normalized distributions."
            >
              <Stage3Scaling dataset={dataset} />
            </StageCardWrapper>
          )}

          {currentStage === 4 && (
            <StageCardWrapper
              stageNumber={4}
              title="Categorical Encoding"
              whatText="Encoding Sex into binary numerical values and splitting Embarked into multiple columns."
              whyText="One-hot encoding represents categories numerically without creating a false order between them."
              resultText="Embarked split into Embarked_C, Embarked_Q, and Embarked_S."
            >
              <Stage4Encoding dataset={dataset} />
            </StageCardWrapper>
          )}

          {currentStage === 5 && (
            <StageCardWrapper
              stageNumber={5}
              title="Create FamilySize"
              whatText="Combining SibSp + Parch + 1 using actual passengers from train.csv."
              whyText="Feature engineering can create new information by combining existing variables."
              resultText="✨ FamilySize — NEW FEATURE added to the dataset."
            >
              <Stage5FamilySize dataset={dataset} />
            </StageCardWrapper>
          )}

          {currentStage === 6 && (
            <StageCardWrapper
              stageNumber={6}
              title="Create IsAlone"
              whatText="Deriving a binary flag indicating whether a passenger is travelling alone."
              whyText="The original dataset did not explicitly contain an “IsAlone” column. We derived this information from existing features. This is a clear example of feature creation."
              resultText="✨ IsAlone — NEW FEATURE added to the dataset."
            >
              <Stage6IsAlone dataset={dataset} />
            </StageCardWrapper>
          )}

          {currentStage === 7 && (
            <StageCardWrapper
              stageNumber={7}
              title="Extract Title from Name"
              whatText="Extracting passenger titles (Mr, Mrs, Miss, Master) programmatically from raw name strings."
              whyText="Useful structured information can sometimes be extracted from raw text features."
              resultText="✨ Title — NEW FEATURE extracted and mapped across the dataset."
            >
              <Stage7TitleExtract dataset={dataset} />
            </StageCardWrapper>
          )}

          {currentStage === 8 && (
            <StageCardWrapper
              stageNumber={8}
              title="Log Transformation of Fare"
              whatText="Applying log(1 + Fare) transformation to compress extreme values."
              whyText="Fare is positively skewed because some passengers paid much higher fares. Log transformation compresses large values and can reduce positive skewness."
              resultText="Skewness significantly reduced, stabilizing the distribution."
            >
              <Stage8LogFare dataset={dataset} />
            </StageCardWrapper>
          )}

          {currentStage === 9 && (
            <StageCardWrapper
              stageNumber={9}
              title="Feature Selection"
              whatText="Choosing final features and pruning identifiers (PassengerId, raw Name, Ticket)."
              whyText="Feature Engineering creates or transforms information, while Feature Selection chooses which available features are retained."
              resultText="Retained high-signal predictors while dropping ungeneralizable text/ids."
            >
              <Stage9Selection />
            </StageCardWrapper>
          )}

          {currentStage === 10 && (
            <StageCardWrapper
              stageNumber={10}
              title="Before vs After"
              whatText="Comparing the raw Titanic dataset side-by-side with ML-ready features."
              whyText="Feature Engineering improves the representation of information supplied to a machine learning model."
              resultText="Visual journey from Raw Data → Meaningful Features → ML-Ready Data."
            >
              <Stage10BeforeAfter />
            </StageCardWrapper>
          )}

          {currentStage === 11 && (
            <StageCardWrapper
              stageNumber={11}
              title="Does Feature Engineering Actually Help?"
              whatText="Evaluating Logistic Regression on basic features vs engineered features."
              whyText="Verifying whether engineered representations improve validation performance without data leakage."
              resultText="Evaluated actual Accuracy, Precision, Recall, and F1 score on holdout validation data."
            >
              <Stage11MLComparison dataset={dataset} />
            </StageCardWrapper>
          )}
        </AnimatePresence>
      </main>

      {/* BOTTOM CONTROLS */}
      <footer className="bottom-controls-bar">
        <div className="controls-inner">
          <button onClick={handlePrev} disabled={currentStage === 1} className="btn-nav">
            <ArrowLeft size={18} /> ← Previous
          </button>

          <button onClick={handleReset} className="btn-reset">
            <RotateCcw size={16} /> ↻ Reset
          </button>

          <button onClick={handleNext} disabled={currentStage === 11} className="btn-nav primary">
            Next → <ArrowRight size={18} />
          </button>
        </div>
      </footer>
    </div>
  );
}
