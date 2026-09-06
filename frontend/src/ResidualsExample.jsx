import { useEffect, useRef, useState } from "react";

const rooms = [1, 2, 4];
const prices = [4500, 19000, 27000];

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
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

// The point with the largest residual (x = 2) is the clearest place to
// anchor the epsilon label - the gap is big enough to point at without
// the tag colliding with the line or the point itself.
const EPSILON_ROOM_INDEX = 1;

function ResidualsExample() {
  const meanX = mean(rooms);
  const meanY = mean(prices);
  const slope =
    rooms.reduce((sum, x, index) => sum + (x - meanX) * (prices[index] - meanY), 0) /
    rooms.reduce((sum, x) => sum + (x - meanX) ** 2, 0);
  const intercept = meanY - slope * meanX;
  const predict = (x) => intercept + slope * x;

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

  const epsilonRoom = rooms[EPSILON_ROOM_INDEX];
  const epsilonActual = prices[EPSILON_ROOM_INDEX];
  const epsilonPredicted = predict(epsilonRoom);
  const epsilonTagX = xScale(epsilonRoom) - 14;
  const epsilonTagY = yScale((epsilonActual + epsilonPredicted) / 2);

  return (
    <svg
      ref={chartRef}
      className={`chart ${isVisible ? "chart-card-visible" : ""}`}
      viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
      role="img"
      aria-label="Apartment price vs. number of rooms, with the model's residual error for each point highlighted"
    >
      <text className="chart-title" x={CHART_WIDTH / 2} y="28" textAnchor="middle">
        The best-fit line still misses every point
      </text>

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

      <line
        className="final-fit-line"
        x1={xScale(X_MIN)}
        y1={yScale(predict(X_MIN))}
        x2={xScale(X_MAX)}
        y2={yScale(predict(X_MAX))}
      />

      {rooms.map((room, index) => (
        <line
          key={`residual-${room}`}
          className="residual-line"
          x1={xScale(room)}
          y1={yScale(prices[index])}
          x2={xScale(room)}
          y2={yScale(predict(room))}
          style={{ transitionDelay: `${1000 + index * 200}ms` }}
        />
      ))}

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

      <g className="term-hover" tabIndex={0}>
        <rect
          className="term-hit-area"
          x={epsilonTagX - 24}
          y={epsilonTagY - 14}
          width="32"
          height="28"
        />

        <text
          className="chart-term-label epsilon-label"
          x={epsilonTagX}
          y={epsilonTagY}
          textAnchor="end"
          dominantBaseline="middle"
        >
          ε
        </text>

        <g
          className="chart-term-tooltip"
          transform={`translate(${epsilonTagX + 20} ${epsilonTagY - 40})`}
        >
          <rect className="tooltip-box" width="230" height="44" />
          <text className="tooltip-text" x="12" y="18">
            The gap between the actual price and
          </text>
          <text className="tooltip-text" x="12" y="34">
            the model&apos;s prediction — the error, ε.
          </text>
        </g>
      </g>
    </svg>
  );
}

export default ResidualsExample;
