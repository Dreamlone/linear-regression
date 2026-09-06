import { useEffect, useMemo, useRef, useState } from "react";

const datasets = [
  {
    id: "rooms-basic",
    type: "model",
    title: "Dataset 1: simple apartment prices",
    points: [
      { x: 1, y: 10000 },
      { x: 2, y: 20000 },
      { x: 4, y: 40000 },
    ],
    targetB0: 0,
    targetB1: 10000,
  },
  {
    id: "rooms-base-price",
    type: "model",
    title: "Dataset 2: apartments with a base price",
    points: [
      { x: 1, y: 15000 },
      { x: 2, y: 25000 },
      { x: 4, y: 45000 },
    ],
    targetB0: 5000,
    targetB1: 10000,
  },
  {
    id: "rooms-negative-slope",
    type: "model",
    title: "Dataset 3: price drops with more rooms?",
    points: [
      { x: 1, y: 13000 },
      { x: 2, y: 11000 },
      { x: 4, y: 7000 },
    ],
    targetB0: 15000,
    targetB1: -2000,
    surpriseNote: "Wow, why is it decreasing?",
  },
  {
    id: "congrats",
    type: "message",
    title: "You've got it",
    heading: "Nice work!",
    body: "You've seen how b₀ shifts the line up and down, and how b₁ controls its slope — including which way it tilts. That's the whole shape of a linear model.",
  },
];

const SLIDER_SNAP_DISTANCE = 750;

function formatFormula(b0, b1) {
  const sign = b1 >= 0 ? "+" : "−";

  return `ŷ = ${b0.toLocaleString()} ${sign} ${Math.abs(
    b1,
  ).toLocaleString()} · x`;
}

function DatasetPlot({
  dataset,
  displayedB0,
  displayedB1,
  isSolved,
  isCurrentDataset,
  width,
  height,
  margin,
  plotWidth,
  plotHeight,
  xMin,
  xMax,
  xScale,
  yScale,
  xTicks,
  yTicks,
}) {
  const modelLine = {
    x1: xMin,
    y1: displayedB0 + displayedB1 * xMin,
    x2: xMax,
    y2: displayedB0 + displayedB1 * xMax,
  };

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Interactive plot for fitting a linear regression model"
    >
      <defs>
        <clipPath id={`plot-clip-${dataset.id}`}>
          <rect
            x={margin.left}
            y={margin.top}
            width={plotWidth}
            height={plotHeight}
          />
        </clipPath>
      </defs>

      <text className="chart-title" x={width / 2} y="28" textAnchor="middle">
        {dataset.title}
      </text>

      {yTicks.map((tick) => (
        <g key={`y-grid-${dataset.id}-${tick}`}>
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
        <g key={`x-tick-${dataset.id}-${tick}`}>
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

      {isCurrentDataset && (
        <g clipPath={`url(#plot-clip-${dataset.id})`}>
          <line
            className={`fit-model-line ${
              isSolved ? "fit-model-line-solved" : ""
            }`}
            x1={xScale(modelLine.x1)}
            y1={yScale(modelLine.y1)}
            x2={xScale(modelLine.x2)}
            y2={yScale(modelLine.y2)}
          />
        </g>
      )}

      {dataset.points.map((point) => (
        <circle
          key={`${dataset.id}-${point.x}-${point.y}`}
          className="data-point"
          cx={xScale(point.x)}
          cy={yScale(point.y)}
          r="4.5"
        />
      ))}
    </svg>
  );
}

function CongratsCard({ dataset }) {
  return (
    <div className="congrats-card">
      <p className="congrats-heading">{dataset.heading}</p>
      <p className="congrats-body">{dataset.body}</p>
    </div>
  );
}

function ModelFittingPlayground() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [dragStartX, setDragStartX] = useState(null);
  const [dragOffsetX, setDragOffsetX] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  const firstCardRef = useRef(null);
  const [cardStep, setCardStep] = useState(0);

  const initialCoefficients = useMemo(
    () =>
      Object.fromEntries(
        datasets.map((dataset) => [
          dataset.id,
          {
            b0: 0,
            b1: 0,
          },
        ]),
      ),
    [],
  );

  const [coefficientsByDataset, setCoefficientsByDataset] =
    useState(initialCoefficients);

  const dataset = datasets[currentIndex];
  const coefficients = coefficientsByDataset[dataset.id];

  // Once the coefficients match the target exactly, updateCoefficient()
  // below refuses further edits, so they stay pinned at the target and
  // this stays true - no separate "solved" state/effect needed.
  const isSolved =
    coefficients.b0 === dataset.targetB0 &&
    coefficients.b1 === dataset.targetB1;

  // Whether the dataset at a given index has been solved - used to gate
  // navigation. Unlike `isSolved` above (which only describes the
  // currently active dataset), this can be checked for any index, so
  // solving dataset N only unlocks dataset N+1, not every later one.
  const isDatasetSolvedAt = (index) => {
    const item = datasets[index];

    if (item.type !== "model") {
      return false;
    }

    const itemCoefficients = coefficientsByDataset[item.id];

    return (
      itemCoefficients.b0 === item.targetB0 &&
      itemCoefficients.b1 === item.targetB1
    );
  };

  const displayedB0 = coefficients.b0;
  const displayedB1 = coefficients.b1;

  const [b0Input, setB0Input] = useState(() => String(coefficients.b0));
  const [b1Input, setB1Input] = useState(() => String(coefficients.b1));


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

  useEffect(() => {
    const updateCardStep = () => {
      if (!firstCardRef.current) {
        return;
      }

      // offsetWidth (layout box), not getBoundingClientRect (visual box) -
      // the first card gets scaled down via CSS transform whenever it isn't
      // the active card, which would otherwise corrupt this measurement.
      const cardWidth = firstCardRef.current.offsetWidth;
      const gap = window.matchMedia("(max-width: 860px)").matches ? 18 : 28;

      setCardStep(cardWidth + gap);
    };

    updateCardStep();

    window.addEventListener("resize", updateCardStep);

    return () => {
      window.removeEventListener("resize", updateCardStep);
    };
  }, []);

  const updateCoefficient = (name, value) => {
    if (isSolved) {
      return;
    }

    if (value === "") {
      return;
    }

    const parsed = Number(value);

    if (Number.isNaN(parsed)) {
      return;
    }

    const clamped = Math.min(Math.max(parsed, -10000), 20000);

    setCoefficientsByDataset((previous) => ({
      ...previous,
      [dataset.id]: {
        ...previous[dataset.id],
        [name]: clamped,
      },
    }));
  };

  const handleSliderChange = (name, value, setLocalText) => {
    const target = name === "b0" ? dataset.targetB0 : dataset.targetB1;
    const raw = Number(value);
    // Magnetic snap: dragging within one and a half steps of the target
    // locks the slider onto it, so you don't have to release on the exact
    // pixel.
    const snapped = Math.abs(raw - target) <= SLIDER_SNAP_DISTANCE ? target : raw;

    updateCoefficient(name, snapped);
    setLocalText(String(snapped));
  };

  const handleNumberInputChange = (name, rawValue, setLocalText) => {
    setLocalText(rawValue);
    updateCoefficient(name, rawValue);
  };

  const goToDataset = (index) => {
    const clampedIndex = Math.min(Math.max(index, 0), datasets.length - 1);
    const nextCoefficients = coefficientsByDataset[datasets[clampedIndex].id];

    setCurrentIndex(clampedIndex);
    setB0Input(String(nextCoefficients.b0));
    setB1Input(String(nextCoefficients.b1));
  };

  const goToPreviousDataset = () => {
    goToDataset(currentIndex - 1);
  };

  const goToNextDataset = () => {
    if (!isSolved) {
      return;
    }

    goToDataset(currentIndex + 1);
  };

  const handlePointerDown = (event) => {
    setDragStartX(event.clientX);
    setDragOffsetX(0);
    setIsDragging(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event) => {
    if (dragStartX === null) {
      return;
    }

    setDragOffsetX(event.clientX - dragStartX);
  };

  const handlePointerUp = () => {
    const swipeThreshold = 72;

    if (dragOffsetX <= -swipeThreshold) {
      goToNextDataset();
    }

    if (dragOffsetX >= swipeThreshold) {
      goToPreviousDataset();
    }

    setDragStartX(null);
    setDragOffsetX(0);
    setIsDragging(false);
  };

  return (
    <div className="model-fitting-playground">
      <h3>Find a model for 3 different datasets</h3>

      <div className="model-fitting-layout">
        <aside className="model-controls">
          {dataset.type === "model" ? (
            <>
              <label>
                <span>b₀, intercept</span>
                <div className="coefficient-row">
                  <input
                    type="range"
                    min="-10000"
                    max="20000"
                    step="500"
                    value={displayedB0}
                    disabled={isSolved}
                    onChange={(event) =>
                      handleSliderChange("b0", event.target.value, setB0Input)
                    }
                  />
                  <input
                    type="number"
                    className="coefficient-input"
                    step="500"
                    value={b0Input}
                    disabled={isSolved}
                    onChange={(event) =>
                      handleNumberInputChange(
                        "b0",
                        event.target.value,
                        setB0Input,
                      )
                    }
                  />
                </div>
              </label>

              <label>
                <span>b₁, slope</span>
                <div className="coefficient-row">
                  <input
                    type="range"
                    min="-10000"
                    max="20000"
                    step="500"
                    value={displayedB1}
                    disabled={isSolved}
                    onChange={(event) =>
                      handleSliderChange("b1", event.target.value, setB1Input)
                    }
                  />
                  <input
                    type="number"
                    className="coefficient-input"
                    step="500"
                    value={b1Input}
                    disabled={isSolved}
                    onChange={(event) =>
                      handleNumberInputChange(
                        "b1",
                        event.target.value,
                        setB1Input,
                      )
                    }
                  />
                </div>
              </label>

              <div className="current-model-card">
                <p>Current model</p>
                <strong>{formatFormula(displayedB0, displayedB1)}</strong>
              </div>

              {isSolved ? (
                <p className="model-status model-status-solved">
                  Model found. The line is fixed.
                </p>
              ) : (
                <p className="model-status">
                  Move the sliders until the line passes through all points.
                </p>
              )}
            </>
          ) : (
            <div className="model-controls-done">
              <p>All three models solved.</p>
              <p className="model-controls-done-sub">
                b₀ and b₁ are all it takes to describe any straight-line
                model.
              </p>
            </div>
          )}
        </aside>

        <section className="model-chart-area">
          <div
            className={`dataset-carousel ${
              isDragging ? "dataset-carousel-dragging" : ""
            }`}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
          >
            <div
              className="dataset-carousel-track"
              style={{
                transform: `translateX(${
                  -(currentIndex * cardStep) + dragOffsetX * 0.45
                }px)`,
              }}
            >
              {datasets.map((item, index) => {
                const distanceFromCurrent = Math.abs(index - currentIndex);
                const isCurrentDataset = index === currentIndex;

                return (
                  <div
                    key={item.id}
                    ref={index === 0 ? firstCardRef : null}
                    className={`dataset-card ${
                      isCurrentDataset ? "dataset-card-active" : ""
                    }`}
                    style={{
                      opacity: distanceFromCurrent === 0 ? 1 : 0.26,
                      transform: `scale(${
                        distanceFromCurrent === 0 ? 1 : 0.82
                      })`,
                    }}
                  >
                    {item.type === "model" ? (
                      <DatasetPlot
                        dataset={item}
                        displayedB0={displayedB0}
                        displayedB1={displayedB1}
                        isSolved={isSolved}
                        isCurrentDataset={isCurrentDataset}
                        width={width}
                        height={height}
                        margin={margin}
                        plotWidth={plotWidth}
                        plotHeight={plotHeight}
                        xMin={xMin}
                        xMax={xMax}
                        yMin={yMin}
                        yMax={yMax}
                        xScale={xScale}
                        yScale={yScale}
                        xTicks={xTicks}
                        yTicks={yTicks}
                      />
                    ) : (
                      <CongratsCard dataset={item} />
                    )}

                    {item.surpriseNote && isCurrentDataset && (
                      <div className="surprise-note">{item.surpriseNote}</div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="dataset-carousel-hint">
            {dataset.type !== "model"
              ? "You've completed every dataset."
              : isSolved
                ? "Drag the chart to move to the next dataset."
                : "Fit the current model first. Then drag to continue."}
          </div>

          <div className="dataset-carousel-dots">
            {datasets.map((item, index) => {
              const isUnlocked = index === 0 || isDatasetSolvedAt(index - 1);

              return (
                <button
                  key={item.id}
                  type="button"
                  className={index === currentIndex ? "active" : ""}
                  disabled={!isUnlocked}
                  onClick={() => {
                    if (isUnlocked) {
                      goToDataset(index);
                    }
                  }}
                  aria-label={item.type === "message" ? "Summary" : `Dataset ${index + 1}`}
                >
                  {item.type === "message" ? "!" : index + 1}
                </button>
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
}

export default ModelFittingPlayground;