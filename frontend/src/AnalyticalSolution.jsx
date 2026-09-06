import { useEffect, useId, useRef, useState } from "react";

const rooms = [1, 2, 4];
const prices = [10000, 20000, 40000];

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function getSlopeBreakdown(xValues, yValues, meanX, meanY) {
  const rows = xValues.map((x, index) => {
    const y = yValues[index];
    const dx = x - meanX;
    const dy = y - meanY;

    return { x, y, dx, dy, product: dx * dy, square: dx * dx };
  });

  const sumProduct = rows.reduce((sum, row) => sum + row.product, 0);
  const sumSquare = rows.reduce((sum, row) => sum + row.square, 0);

  return { rows, sumProduct, sumSquare, slope: sumProduct / sumSquare };
}

function formatSigned(value, decimals) {
  const formatted = value.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

  return formatted.replace(/^-/, "−");
}

const CHART_WIDTH = 560;
const CHART_HEIGHT = 360;
const CHART_MARGIN = { top: 52, right: 28, bottom: 72, left: 86 };
const PLOT_WIDTH = CHART_WIDTH - CHART_MARGIN.left - CHART_MARGIN.right;
const PLOT_HEIGHT = CHART_HEIGHT - CHART_MARGIN.top - CHART_MARGIN.bottom;
const X_MIN = 0;
const X_MAX = 5;
const Y_MIN = 0;
const Y_MAX = 50000;
const X_TICKS = [0, 1, 2, 3, 4, 5];
const Y_TICKS = [0, 10000, 20000, 30000, 40000, 50000];

function xScale(x) {
  return CHART_MARGIN.left + ((x - X_MIN) / (X_MAX - X_MIN)) * PLOT_WIDTH;
}

function yScale(y) {
  return (
    CHART_MARGIN.top +
    PLOT_HEIGHT -
    ((y - Y_MIN) / (Y_MAX - Y_MIN)) * PLOT_HEIGHT
  );
}

const LABEL_X = 2.9;

function MeanFitChart({
  showMean,
  showLine,
  finalLine,
  meanX,
  meanY,
  slope,
  intercept,
  altSlope,
  altIntercept,
  predictions,
}) {
  // The "correct" line always passes through the mean point by definition,
  // so when an explicit intercept isn't given (level 2, before b0 exists
  // yet), it's derived from the slope and the mean instead - same line,
  // just not yet named "the intercept".
  const correctB0 = intercept !== undefined ? intercept : meanY - slope * meanX;
  const correctY = (x) => correctB0 + slope * x;

  const hasAlt = altSlope !== undefined || altIntercept !== undefined;
  const altB1 = altSlope !== undefined ? altSlope : slope;
  const altB0 =
    altIntercept !== undefined ? altIntercept : meanY - altSlope * meanX;
  const altY = (x) => altB0 + altB1 * x;

  const line = showLine
    ? { x1: X_MIN, y1: correctY(X_MIN), x2: X_MAX, y2: correctY(X_MAX) }
    : null;

  const altLine =
    showLine && hasAlt
      ? { x1: X_MIN, y1: altY(X_MIN), x2: X_MAX, y2: altY(X_MAX) }
      : null;

  const clipId = useId();

  const chartRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.35 },
    );

    if (chartRef.current) {
      observer.observe(chartRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <svg
      ref={chartRef}
      className={`chart ${isVisible ? "chart-card-visible" : ""}`}
      viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
      role="img"
      aria-label="Apartment price vs. number of rooms, with the mean of each variable highlighted"
    >
      <defs>
        <clipPath id={clipId}>
          <rect
            x={CHART_MARGIN.left}
            y={CHART_MARGIN.top}
            width={PLOT_WIDTH}
            height={PLOT_HEIGHT}
          />
        </clipPath>
      </defs>

      {Y_TICKS.map((tick) => (
        <g key={`y-grid-${tick}`}>
          <line
            className="grid-line"
            x1={CHART_MARGIN.left}
            y1={yScale(tick)}
            x2={CHART_WIDTH - CHART_MARGIN.right}
            y2={yScale(tick)}
          />
          <text
            className="tick-label"
            x={CHART_MARGIN.left - 12}
            y={yScale(tick)}
            textAnchor="end"
            dominantBaseline="middle"
          >
            {tick.toLocaleString()}
          </text>
        </g>
      ))}

      {X_TICKS.map((tick) => (
        <g key={`x-tick-${tick}`}>
          <line
            className="tick-line"
            x1={xScale(tick)}
            y1={CHART_MARGIN.top + PLOT_HEIGHT}
            x2={xScale(tick)}
            y2={CHART_MARGIN.top + PLOT_HEIGHT + 7}
          />
          <text
            className="tick-label"
            x={xScale(tick)}
            y={CHART_MARGIN.top + PLOT_HEIGHT + 28}
            textAnchor="middle"
          >
            {tick}
          </text>
        </g>
      ))}

      <line
        className="axis-line"
        x1={CHART_MARGIN.left}
        y1={CHART_MARGIN.top + PLOT_HEIGHT}
        x2={CHART_WIDTH - CHART_MARGIN.right}
        y2={CHART_MARGIN.top + PLOT_HEIGHT}
      />

      <line
        className="axis-line"
        x1={CHART_MARGIN.left}
        y1={CHART_MARGIN.top}
        x2={CHART_MARGIN.left}
        y2={CHART_MARGIN.top + PLOT_HEIGHT}
      />

      <text
        className="axis-label"
        x={CHART_MARGIN.left + PLOT_WIDTH / 2}
        y={CHART_HEIGHT - 20}
        textAnchor="middle"
      >
        Number of rooms
      </text>

      <text
        className="axis-label"
        transform={`translate(24 ${CHART_MARGIN.top + PLOT_HEIGHT / 2}) rotate(-90)`}
        textAnchor="middle"
      >
        Price, $
      </text>

      <g clipPath={`url(#${clipId})`}>
        {altLine && (
          <line
            className="alt-fit-line"
            x1={xScale(altLine.x1)}
            y1={yScale(altLine.y1)}
            x2={xScale(altLine.x2)}
            y2={yScale(altLine.y2)}
          />
        )}

        {line && (
          <line
            className={finalLine ? "final-fit-line" : "mean-fit-line"}
            x1={xScale(line.x1)}
            y1={yScale(line.y1)}
            x2={xScale(line.x2)}
            y2={yScale(line.y2)}
          />
        )}
      </g>

      {altLine && (
        <text
          className="chart-line-label alt-line-label"
          x={xScale(LABEL_X) + 8}
          y={yScale(altY(LABEL_X)) + 16}
        >
          this one?
        </text>
      )}

      {line && altLine && (
        <text
          className="chart-line-label correct-line-label"
          x={xScale(LABEL_X) + 8}
          y={yScale(correctY(LABEL_X)) - 8}
        >
          or that one?
        </text>
      )}

      {rooms.map((room, index) => (
        <circle
          key={`raw-${room}`}
          className="raw-data-point animated-data-point"
          cx={xScale(room)}
          cy={yScale(prices[index])}
          r="4.5"
          style={{ transitionDelay: `${index * 300}ms` }}
        />
      ))}

      {showMean && (
        <circle
          className="mean-point animated-data-point"
          cx={xScale(meanX)}
          cy={yScale(meanY)}
          r="5.5"
          style={{ transitionDelay: `${rooms.length * 300 + 200}ms` }}
        />
      )}

      {predictions &&
        predictions.map((prediction) => {
          const px = xScale(prediction.x);
          const py = yScale(correctY(prediction.x));
          // Label sits above and to the left of its point: anchoring the
          // text at its end (rather than its start) keeps it clear of the
          // point and the line regardless of how long the word is.
          const labelX = px - 10;
          const labelY = py - 12;

          return (
            <g key={prediction.term} className="term-hover" tabIndex={0}>
              <circle
                className="predicted-point animated-data-point"
                cx={px}
                cy={py}
                r="5.5"
                style={{ transitionDelay: "1100ms" }}
              />

              <rect
                className="term-hit-area"
                x={labelX - 105}
                y={labelY - 14}
                width="110"
                height="20"
              />

              <text
                className="chart-term-label"
                x={labelX}
                y={labelY}
                textAnchor="end"
              >
                {prediction.term}
              </text>

              <g
                className="chart-term-tooltip"
                transform={`translate(${px + prediction.tooltipDx} ${
                  py + prediction.tooltipDy
                })`}
              >
                <rect className="tooltip-box" width="190" height="44" />
                <text className="tooltip-text" x="12" y="18">
                  {prediction.description[0]}
                </text>
                <text className="tooltip-text" x="12" y="34">
                  {prediction.description[1]}
                </text>
              </g>
            </g>
          );
        })}
    </svg>
  );
}

function NumberCube({ label, highlight }) {
  return (
    <div className={`number-cube ${highlight ? "number-cube-highlight" : ""}`}>
      {label}
    </div>
  );
}

const ALT_SLOPE = 4000;
const ALT_INTERCEPT = -6000;

const PREDICTIONS = [
  {
    x: 3,
    term: "interpolation",
    description: [
      "Predicting inside the range",
      "of the observed data.",
    ],
    tooltipDx: -195,
    tooltipDy: -70,
  },
  {
    x: 5,
    term: "extrapolation",
    description: [
      "Predicting beyond the observed",
      "range — less reliable.",
    ],
    tooltipDx: -195,
    tooltipDy: 20,
  },
];

function AnalyticalSolution() {
  const meanX = mean(rooms);
  const meanY = mean(prices);
  const { rows, sumProduct, sumSquare, slope } = getSlopeBreakdown(
    rooms,
    prices,
    meanX,
    meanY,
  );
  const intercept = meanY - slope * meanX;

  return (
    <div className="analytical-solution">
      <h3 className="analytical-solution-title">
        Analytical solution. Simple regression with one feature
      </h3>

      <div className="analytical-level">
        <div className="analytical-level-heading">
          <span className="analytical-level-number">1</span>
          <h4>Find mean values</h4>
        </div>

        <div className="analytical-level-body">
          <div className="analytical-level-left">
            <p className="analytical-row-label">Number of rooms, x</p>
            <div className="number-cube-row">
              {rooms.map((value) => (
                <NumberCube key={`x-${value}`} label={value} />
              ))}
              <span className="number-cube-arrow">→</span>
              <NumberCube label={`x̄ ≈ ${meanX.toFixed(2)}`} highlight />
            </div>

            <p className="analytical-row-label">Price, y</p>
            <div className="number-cube-row">
              {prices.map((value) => (
                <NumberCube key={`y-${value}`} label={value.toLocaleString()} />
              ))}
              <span className="number-cube-arrow">→</span>
              <NumberCube
                label={`ȳ ≈ ${meanY.toLocaleString(undefined, {
                  maximumFractionDigits: 2,
                })}`}
                highlight
              />
            </div>
          </div>

          <div className="analytical-level-right">
            <MeanFitChart showMean meanX={meanX} meanY={meanY} />
          </div>
        </div>
      </div>

      <div className="analytical-level">
        <div className="analytical-level-heading">
          <span className="analytical-level-number">2</span>
          <h4>Compute slope coefficient b₁</h4>
        </div>

        <div className="analytical-level-body">
          <div className="analytical-level-left">
            <div className="analytical-formula">
              <span className="analytical-formula-main">
                b₁ = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)²
              </span>

              <div className="slope-table-wrapper">
                <table className="slope-table">
                  <thead>
                    <tr>
                      <th>x</th>
                      <th>y</th>
                      <th>x − x̄</th>
                      <th>y − ȳ</th>
                      <th>(x−x̄)(y−ȳ)</th>
                      <th>(x−x̄)²</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => (
                      <tr key={row.x}>
                        <td>{row.x}</td>
                        <td>{row.y.toLocaleString()}</td>
                        <td>{formatSigned(row.dx, 4)}</td>
                        <td>{formatSigned(row.dy, 2)}</td>
                        <td>{formatSigned(row.product, 2)}</td>
                        <td>{row.square.toFixed(4)}</td>
                      </tr>
                    ))}
                    <tr className="slope-table-total">
                      <td colSpan={4}>Σ</td>
                      <td>{formatSigned(sumProduct, 2)}</td>
                      <td>{sumSquare.toFixed(4)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <span className="analytical-formula-step">
                b₁ = {formatSigned(sumProduct, 2)} / {sumSquare.toFixed(4)}
              </span>

              <span className="analytical-formula-result">
                b₁ ≈ {Math.round(slope).toLocaleString()}
              </span>
            </div>
          </div>

          <div className="analytical-level-right">
            <MeanFitChart
              showMean
              showLine
              meanX={meanX}
              meanY={meanY}
              slope={slope}
              altSlope={ALT_SLOPE}
            />
          </div>
        </div>
      </div>

      <div className="analytical-level">
        <div className="analytical-level-heading">
          <span className="analytical-level-number">3</span>
          <h4>Compute the intercept b₀</h4>
        </div>

        <div className="analytical-level-body">
          <div className="analytical-level-left">
            <div className="analytical-formula">
              <span className="analytical-formula-main">b₀ = ȳ − b₁ · x̄</span>

              <span className="analytical-formula-step">
                b₀ = {formatSigned(meanY, 2)} − {Math.round(slope).toLocaleString()} ×{" "}
                {meanX.toFixed(4)}
              </span>

              <span className="analytical-formula-step">
                b₀ = {formatSigned(meanY, 2)} − {formatSigned(slope * meanX, 2)}
              </span>

              <span className="analytical-formula-result">
                b₀ ≈ {(Math.round(intercept) || 0).toLocaleString()}
              </span>
            </div>
          </div>

          <div className="analytical-level-right">
            <MeanFitChart
              showMean
              showLine
              meanX={meanX}
              meanY={meanY}
              slope={slope}
              intercept={intercept}
              altIntercept={ALT_INTERCEPT}
            />
          </div>
        </div>
      </div>

      <div className="analytical-level">
        <div className="analytical-level-heading">
          <span className="analytical-level-number">4</span>
          <h4>Substitute estimated coefficients</h4>
        </div>

        <div className="analytical-level-body">
          <div className="analytical-level-left">
            <div className="analytical-formula">
              <span className="analytical-formula-main">ŷ = b₀ + b₁ · x</span>

              <span className="analytical-formula-result">
                ŷ = {(Math.round(intercept) || 0).toLocaleString()} +{" "}
                {Math.round(slope).toLocaleString()} · x
              </span>

              <span className="analytical-formula-step">
                The model is complete. Hover the labels on the chart to see
                what it can — and can't — safely predict.
              </span>
            </div>
          </div>

          <div className="analytical-level-right">
            <MeanFitChart
              showMean
              showLine
              finalLine
              meanX={meanX}
              meanY={meanY}
              slope={slope}
              intercept={intercept}
              predictions={PREDICTIONS}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default AnalyticalSolution;
