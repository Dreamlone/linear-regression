import { useEffect, useMemo, useRef, useState } from "react";

function ApartmentScatterPlot() {
  const chartRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);

  const rooms = [1, 2, 4];
  const prices = [10000, 20000, 40000];

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
      rooms
        .map((room, index) => ({
          room,
          price: prices[index],
        }))
        .sort((a, b) => b.price - a.price),
    [],
  );

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

  return (
    <figure
      ref={chartRef}
      className={`chart-card ${isVisible ? "chart-card-visible" : ""}`}
    >
      <svg
        className="chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Scatter plot showing apartment price versus number of rooms"
      >
        <text className="chart-title" x={width / 2} y="28" textAnchor="middle">
          Apartment price vs. number of rooms
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

        {points.map((point, index) => (
          <circle
            key={point.room}
            className="data-point animated-data-point"
            cx={xScale(point.room)}
            cy={yScale(point.price)}
            r="4.5"
            style={{
              transitionDelay: `${index * 300}ms`,
            }}
          />
        ))}
      </svg>
    </figure>
  );
}

export default ApartmentScatterPlot;