export default function ConfidenceGauge({ value, size = 80 }) {
  const pct = Math.round(value * 100)
  const r = (size - 8) / 2
  const circ = 2 * Math.PI * r
  const offset = circ * (1 - value)
  const color = pct >= 80 ? '#34d399' : pct >= 60 ? '#fbbf24' : '#f87171'

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size/2} cy={size/2} r={r}
        fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="6" />
      <circle cx={size/2} cy={size/2} r={r}
        className="gauge-ring" stroke={color} strokeWidth="6"
        strokeDasharray={circ} strokeDashoffset={offset}
        transform={`rotate(-90 ${size/2} ${size/2})`} />
      <text x={size/2} y={size/2} textAnchor="middle" dominantBaseline="central"
        fill={color} fontSize="18" fontWeight="800">
        {pct}
      </text>
    </svg>
  )
}
