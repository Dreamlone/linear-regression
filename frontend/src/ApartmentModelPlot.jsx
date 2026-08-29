import { useEffect, useMemo, useRef, useState } from "react";

function ApartmentModelPlot() {
  const chartRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);
  const [tooltip, setTooltip] = useState(null);

  const rooms = [1, 2, 4];
  const prices = [10000, 20000, 40000];

  const model = {
    b0: 0,
    b1: 10000,
  };

  const width = 560;
  const height = 360;

  const margin = {
    top: 52,
    right: 28,
    bottom: 72,
    left: 86,
  };

  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const xMin = 0;
  const xMax = 5;
  const yMin = 0;
  const yMax = 50000;

  const xScale = (x) =>
    margin.left + ((x - xMin) / (xMax - xMin)) * plotWidth;

  const yScale = (y) =>
    margin.top + plotHeight - ((y - yMin) / (yMax - yMin)) * plotHeight;

  const xTicks = [0, 1, 2, 3, 4, 5];
  const yTicks = [0, 10000, 20000, 30000, 40000, 50000];

  const points = useMemo(
    () =>
      rooms.map((room, index) => ({
        room,
        price: prices[index],
      })),
    [],
  );

  const modelLine = {
    x1: 0,
    y1: model.b0 + model.b1 * 0,
    x2: 5,
    y2: model.b0 + model.b1 * 5,
  };

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      {
        threshold: 0.35,
      },
    );

    if (chartRef.current) {
      observer.observe(chartRef.current);
    }

    return () => observer.disconnect();
  }, []);

  const handleModelMouseMove = (event) => {
    const svg = event.currentTarget.ownerSVGElement;
    const rect = svg.getBoundingClientRect();
    const viewBoxScaleX = width / rect.width;

    const mouseX = (event.clientX - rect.left) * viewBoxScaleX;
    const xValue =
      xMin + ((mouseX - margin.left) / plotWidth) * (xMax - xMin);

    const clippedXValue = Math.min(Math.max(xValue, xMin), xMax);
    const prediction = model.b0 + model.b1 * clippedXValue;

    setTooltip({
      x: xScale(clippedXValue),
      y: yScale(prediction),
      room: clippedXValue,
      prediction,
    });
  };

  return (
    <figure
      ref={chartRef}
      className={`chart-card ${isVisible ? "chart-card-visible" : ""}`}
    >
      <svg
        className="chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Scatter plot showing apartment price versus number of rooms with a fitted linear regression model"
        onMouseLeave={() => setTooltip(null)}
      >
        <text className="chart-title" x={width / 2} y="28" textAnchor="middle">
          Data-driven model: ŷ = b₀ + b₁ · x = 0 + 10000 · x
        </text>

        {yTicks.map((tick) => (
          <g key={`y-grid-${tick}`}>
            <line
              className="grid-line"
              x1={margin.left}
              y1={yScale(tick)}
              x2={width - margin.right}
              y2={yScale(tick)}
            />
            <text
              className="tick-label"
              x={margin.left - 12}
              y={yScale(tick)}
              textAnchor="end"
              dominantBaseline="middle"
            >
              {tick.toLocaleString()}
            </text>
          </g>
        ))}

        {xTicks.map((tick) => (
          <g key={`x-tick-${tick}`}>
            <line
              className="tick-line"
              x1={xScale(tick)}
              y1={margin.top + plotHeight}
              x2={xScale(tick)}
              y2={margin.top + plotHeight + 7}
            />
            <text
              className="tick-label"
              x={xScale(tick)}
              y={margin.top + plotHeight + 28}
              textAnchor="middle"
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
          y={height - 20}
          textAnchor="middle"
        >
          Number of rooms
        </text>

        <text
          className="axis-label"
          transform={`translate(24 ${margin.top + plotHeight / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          Price, $
        </text>

        <line
          className="model-line"
          x1={xScale(modelLine.x1)}
          y1={yScale(modelLine.y1)}
          x2={xScale(modelLine.x2)}
          y2={yScale(modelLine.y2)}
        />

        <line
          className="model-line-hitbox"
          x1={xScale(modelLine.x1)}
          y1={yScale(modelLine.y1)}
          x2={xScale(modelLine.x2)}
          y2={yScale(modelLine.y2)}
          onMouseMove={handleModelMouseMove}
        />

        {points.map((point) => (
          <circle
            key={point.room}
            className="data-point"
            cx={xScale(point.room)}
            cy={yScale(point.price)}
            r="4.5"
          />
        ))}

        {tooltip && (
          <g className="model-tooltip">
            <line
              className="tooltip-guide"
              x1={tooltip.x}
              y1={tooltip.y}
              x2={tooltip.x}
              y2={margin.top + plotHeight}
            />

            <circle
              className="tooltip-point"
              cx={tooltip.x}
              cy={tooltip.y}
              r="4"
            />

            <g
              transform={`translate(${Math.min(
                tooltip.x + 12,
                width - margin.right - 150,
              )} ${Math.max(tooltip.y - 46, margin.top + 10)})`}
            >
              <rect className="tooltip-box" width="138" height="42" />
              <text className="tooltip-text" x="10" y="17">
                x = {tooltip.room.toFixed(2)}
              </text>
              <text className="tooltip-text" x="10" y="33">
                ŷ = ${Math.round(tooltip.prediction).toLocaleString()}
              </text>
            </g>
          </g>
        )}
      </svg>
    </figure>
  );
}

export default ApartmentModelPlot;