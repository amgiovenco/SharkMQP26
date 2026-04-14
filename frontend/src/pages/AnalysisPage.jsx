import { useRef, useState } from 'react'
import ResultCard from '../components/ResultCard'

const LogoShark = () => (
  <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
    <path d="M4,32 C10,18 26,12 40,14 C52,16 60,24 60,32 C54,40 44,44 32,42 C18,40 6,38 4,32 Z" />
    <path d="M38,6 L46,20 L34,20 Z" />
    <path d="M56,28 L64,22 L62,40 Z" />
  </svg>
)

export default function AnalysisPage({ token, onLogout }) {
  const [file, setFile]         = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [loading, setLoading]   = useState(false)
  const [results, setResults]   = useState(null)
  const [error, setError]       = useState('')
  const inputRef = useRef()

  /* ── file handling ── */
  const acceptFile = (f) => {
    if (!f) return
    if (!f.name.match(/\.csv$/i)) {
      setError('Please upload a .csv file.')
      return
    }
    setFile(f)
    setResults(null)
    setError('')
  }

  const onDrop = (e) => {
    e.preventDefault(); setDragOver(false)
    acceptFile(e.dataTransfer.files[0])
  }

  /* ── analyze ── */
  const analyze = async () => {
    if (!file) return
    setLoading(true); setError(''); setResults(null)

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      })

      if (res.status === 401) { onLogout(); return }

      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Analysis failed')
      setResults(data.results)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  /* ── download CSV ── */
  const downloadCSV = () => {
    if (!results) return
    const header = 'Sample,Rank,Species,Confidence'
    const rows = results.flatMap(r =>
      (r.predictions || []).map(p =>
        `${r.sample_index + 1},${p.rank},"${p.species}",${(p.confidence * 100).toFixed(2)}%`
      )
    )
    const blob = new Blob([header + '\n' + rows.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'wildsense_results.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  /* ── render ── */
  return (
    <>
      <div className="ocean-bg" />

      <div className="analysis-page">
        {/* navbar */}
        <nav className="navbar">
          <div className="nav-brand">
            <LogoShark />
            WildSense
          </div>
          <button className="btn btn-danger" onClick={onLogout}>
            Sign out
          </button>
        </nav>

        <div className="main">
          <h1 className="page-heading">Species Analysis</h1>
          <p className="page-desc">
            Upload a qPCR melting curve CSV — raw or pre-processed. All samples in the file will be identified.
          </p>

          {/* drop zone */}
          <div
            className={`upload-zone${dragOver ? ' drag-over' : ''}${file ? ' has-file' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={e => acceptFile(e.target.files[0])}
            />
            <div className="upload-icon">
              {file ? '📄' : '🌊'}
            </div>
            {file ? (
              <>
                <p className="upload-title">File selected</p>
                <div className="file-pill">
                  <span>📎</span>
                  <span>{file.name}</span>
                  <span style={{ opacity: .6 }}>({(file.size / 1024).toFixed(1)} KB)</span>
                </div>
              </>
            ) : (
              <>
                <p className="upload-title">Drop your CSV here</p>
                <p className="upload-hint">or click to browse — .csv only</p>
              </>
            )}
          </div>

          {/* action row */}
          <div className="analyze-row">
            <button
              className="btn btn-primary"
              onClick={analyze}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2, margin: 0 }} />
                  Analyzing…
                </>
              ) : '🔬 Run Analysis'}
            </button>

            {file && !loading && (
              <button
                className="btn btn-outline"
                onClick={() => { setFile(null); setResults(null); setError('') }}
              >
                Clear
              </button>
            )}
          </div>

          {/* error */}
          {error && <div className="error-banner">⚠ {error}</div>}

          {/* loading */}
          {loading && (
            <div className="loading">
              <div className="spinner" />
              <p className="loading-text">Running ResNet-18 inference on all samples…</p>
            </div>
          )}

          {/* results */}
          {results && !loading && (
            <>
              <div className="results-bar">
                <h2 style={{ fontSize: '1.15rem', fontWeight: 700 }}>Results</h2>
                <span className="results-label">{results.length} sample{results.length !== 1 ? 's' : ''} processed</span>
                <button className="btn btn-outline" onClick={downloadCSV}>
                  Download CSV
                </button>
              </div>

              {results.length === 0 && (
                <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--muted)' }}>
                  No samples found in this file.
                </div>
              )}

              {results.map(r => (
                <ResultCard key={r.sample_index} result={r} />
              ))}
            </>
          )}
        </div>
      </div>
    </>
  )
}
