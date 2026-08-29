import "./App.css";
import ApartmentScatterPlot from "./ApartmentScatterPlot.jsx";
import ApartmentModelPlot from "./ApartmentModelPlot.jsx";
import ModelFittingPlayground from "./ModelFittingPlayground.jsx";

function App() {
  return (
    <main className="page">
      <section className="hero">
        <p className="eyebrow">Interactive visual guide</p>

        <h1>A Visual Explanation of Linear Regression</h1>

        <div className="sections">
          <article>
            <span className="section-number">01</span>
            <h2>What it is and why we use it</h2>
            <p>
              We start with the basic idea of a model: a simple mathematical
              relationship that helps us describe patterns in data and make
              predictions.
            </p>

            <ApartmentScatterPlot />

            <div className="formula-explanation">
              <p>
                In linear regression, we model linear relationships between data
                variables. In simple one-feature regression, when there is one
                feature and one target variable, the equation has the form:
              </p>

              <div className="formula-block">
                <span className="main-formula">y = b₀ + b₁ · x</span>
              </div>

              <p>
                where <strong>x</strong> is the feature, <strong>y</strong> is
                the target variable. [James, G., et al. Linear Regression. An
                Introduction to Statistical Learning, 2021. Free version:{" "}
                <a
                  href="https://www.statlearning.com/"
                  target="_blank"
                  rel="noreferrer"
                >
                  https://www.statlearning.com/
                </a>
                ]
              </p>

              <p>
                So the expression{" "}
                <span className="inline-formula">y = 1 + 10 · x</span>{" "}
                is a linear regression model. And{" "}
                <span className="inline-formula">y = 15 − 21 · x</span>{" "}
                is one as well. The only difference is the coefficients. Since
                the coefficients are the key parameters of the equation, they
                have their own names:
              </p>

              <ul className="coefficient-list">
                <li>
                  <strong>b₀</strong> is the intercept, also called the bias term.
                </li>
                <li>
                  <strong>b₁</strong> is the slope coefficient.
                </li>
              </ul>
            </div>

            <ApartmentModelPlot />

            <ModelFittingPlayground />
          </article>

          <article>
            <span className="section-number">02</span>
            <h2>How to fit a model</h2>
            <p>
              Then we look at how a regression line is chosen, what its
              coefficients mean, and why minimizing errors leads us to the best
              fit.
            </p>
          </article>

          <article>
            <span className="section-number">03</span>
            <h2>How to evaluate model quality</h2>
            <p>
              A model should not only look reasonable. We also need metrics,
              residuals, validation, and uncertainty estimates to understand its
              quality.
            </p>
          </article>

          <article>
            <span className="section-number">04</span>
            <h2>How to improve the model when the results are not good enough</h2>
            <p>
              Finally, we explore how better data, better features, diagnostics,
              optimization, and regularization can improve model performance.
            </p>
          </article>
        </div>
      </section>
    </main>
  );
}

export default App;