const rooms = [1, 2, 4];

// A stub for the "more than one feature" figure - not decided yet how the
// full interactive version will work, so this only sets up the contrast
// the article text refers to: a feature that carries new information
// versus one that is just a rescaled copy of x1 (collinearity).
function MiniFeatureChart({ yValues, yMax, showLine, slope }) {
  const width = 260;
  const height = 200;
  const margin = { top: 16, right: 16, bottom: 40, left: 52 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xMax = 5;

  const xScale = (x) => margin.left + (x / xMax) * plotWidth;
  const yScale = (y) => margin.top + plotHeight - (y / yMax) * plotHeight;

  const yTicks = [0, yMax / 2, yMax];

  return (
    <svg
      className="chart mini-feature-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Placeholder chart comparing a second feature to the number of rooms"
    >
      {yTicks.map((tick) => (
        <g key={`y-${tick}`}>
          <line
            className="grid-line"
            x1={margin.left}
            y1={yScale(tick)}
            x2={width - margin.right}
            y2={yScale(tick)}
          />
          <text
            className="tick-label"
            x={margin.left - 8}
            y={yScale(tick)}
            textAnchor="end"
            dominantBaseline="middle"
          >
            {tick}
          </text>
        </g>
      ))}

      <line
        className="axis-line"
        x1={margin.left}
        y1={margin.top + plotHeight}
        x2={width - margin.right}
        y2={margin.top + plotHeight}
      />
      <line
        className="axis-line"
        x1={margin.left}
        y1={margin.top}
        x2={margin.left}
        y2={margin.top + plotHeight}
      />

      <text
        className="axis-label"
        x={margin.left + plotWidth / 2}
        y={height - 10}
        textAnchor="middle"
      >
        x₁ (rooms)
      </text>
      <text
        className="axis-label"
        transform={`translate(10 ${margin.top + plotHeight / 2}) rotate(-90)`}
        textAnchor="middle"
      >
        x₂
      </text>

      {showLine && (
        <line
          className="placeholder-trend-line"
          x1={xScale(0)}
          y1={yScale(0)}
          x2={xScale(xMax)}
          y2={yScale(slope * xMax)}
        />
      )}

      {rooms.map((room, index) => (
        <circle
          key={`pt-${room}`}
          className="raw-data-point"
          cx={xScale(room)}
          cy={yScale(yValues[index])}
          r="4"
        />
      ))}
    </svg>
  );
}

function MultipleRegressionPlaceholder() {
  return (
    <div className="multiple-regression-placeholder">
      <h3 className="analytical-solution-title">
        Multiple linear regression (when we have 2 features and more)
      </h3>

      <div className="placeholder-columns">
        <div className="placeholder-column">
          <p className="analytical-row-label">x₂ independent of x₁</p>
          <MiniFeatureChart yValues={[3, 1, 4]} yMax={5} />
        </div>

        <div className="placeholder-column">
          <p className="analytical-row-label">x₂ linearly dependent on x₁</p>
          <MiniFeatureChart
            yValues={rooms.map((room) => room * 2)}
            yMax={10}
            showLine
            slope={2}
          />
        </div>
      </div>
    </div>
  );
}

export default MultipleRegressionPlaceholder;
