# Interactive Playground Development Guide

## Scope

These instructions apply primarily to the interactive application located in:

`frontend/`

The application is published as part of the Linear Regression project at:

`/linear-regression/interactive/`

The Quarto articles under `site/` are separate and should not be modified unless the task explicitly requires it.

---

# Main Goal

The goal of the interactive playground is NOT to contain as much educational material as possible.

The priority is:

> A small number of polished, intuitive, visually appealing and smooth interactive explanations.

Quality is more important than quantity.

It should feel pleasant to explore.

Prefer:

- 3 excellent interactive demonstrations

over:

- 15 mediocre demonstrations.

Do not add content simply because a concept exists in the article.

---

# First Step: Understand the Existing Application

Before implementing significant changes:

1. Inspect the current `frontend/` implementation.
2. Understand the component structure and application flow.
3. Identify existing interactive demonstrations.
4. Identify duplicated or unnecessarily complex code.
5. Identify visual inconsistencies.
6. Identify obvious performance problems.
7. Understand how animations and visualizations are implemented.
8. Check the existing Vite/build configuration.
9. Run the application locally and inspect the actual UI before redesigning it.

Do not immediately rewrite the application from scratch.

Prefer incremental improvement unless the current implementation clearly prevents it.

---

# Product Philosophy

The playground complements the article.

It should help the user develop intuition by manipulating something and immediately seeing the result.

Good interaction:

`change parameter → see consequence`

For example:

`change b0/b1 → regression line moves`

`move a data point → fitted line changes`

`add an outlier → metrics and model react`

The user should understand what an interaction does without reading a long explanation.

Avoid turning the playground into another textbook.

Keep explanatory text short.

The article contains the detailed explanation.
The playground provides experimentation and intuition.

---

# Section Identity and Narrative Progression

The playground follows the article's four-part structure:

1. What it is and why we use it
2. How to fit a model
3. How to evaluate model quality
4. How to improve the model when the results are not good enough

Each block should feel distinct within one shared visual language (see "UX and Visual Design"). That distinctness comes from three separate layers - it is not achieved by changing the base design system per block.

## 1. Visual motif

The additional imagery a block introduces, layered on top of the shared x/y chart language every block keeps using.

- Block 1: the existing x/y scatter plots and model line. This is the baseline every other block builds on.
- Block 2: adds vectors, matrix blocks, and "number cubes" representing the entries of a matrix, as a layer on top of the existing charts - not a replacement for them.
- Blocks 3 and 4: not decided yet. Do not invent a motif preemptively - revisit when work on that block starts.

## 2. Game-genre reference

A loose thematic reference point for tone and feel, not a literal reskin.

- Block 1: casual games.
- Block 2: survival horror (concept only for now - the concrete visual treatment is still undecided).
- Blocks 3 and 4: undecided. Leave them alone until we return to them.

## 3. Interaction idea

What the reader manipulates, and what role they play.

- Block 1 ("what it is"): find the right coefficients by hand, see the line match the points. Role: a newcomer touching things for the first time.
- Block 2 ("how it's fit"): the mechanic gains one dimension of depth - not "find the ready answer," but "watch the algorithm find it step by step."
- Block 3 ("evaluating quality"): a "find the problem" mechanic - show the reader several models or fits and have them diagnose what's wrong.
- Block 4 ("improving the model"): an open sandbox with levers the reader can pull, with a visible, live consequence.

## How These Layers Combine

The mapping between a block's game-genre reference, its visual motif, and its interaction idea is not always obvious up front - that is expected, not a problem to solve immediately.

Do not force a literal, on-the-nose translation of a genre reference (for example, "survival horror" does not necessarily mean literal horror-game UI chrome). When it is unclear how a genre reference should manifest concretely, ask rather than guessing or inventing detail that has not been decided yet.

Visual "sugar" (hover feedback, drag response, satisfying micro-animations) should be shaped by each block's specific mechanic and genre reference, not generic decoration copied between blocks. What makes an interaction feel good in block 1 is not necessarily what makes it feel good in block 2.

---

# UX and Visual Design

Treat visual quality as a first-class requirement.

The interface should be:

- clean
- calm
- modern
- visually consistent
- easy to understand
- pleasant to spend time in
- usable without instructions whenever possible

Prefer whitespace and clear hierarchy over dense interfaces.

Avoid:

- excessive cards
- excessive borders
- excessive gradients
- unnecessary shadows
- decorative UI that does not communicate anything
- dashboards full of controls
- displaying every possible parameter
- large blocks of explanatory text
- inconsistent typography
- arbitrary colors

Every visible element should have a purpose.

Controls should be placed close to the visualization they affect.

Important state changes should be visually obvious.

Keep the visualization itself as the main focus.

## Color System

The palette is neutral (calm, warm background and text) plus a single accent color: orange.

Use orange deliberately, for things that matter, not as decoration:

- the active or selected state
- a solved/success state
- the value currently being manipulated
- an important highlighted formula or interactive frame

Important interactive frames - mini-games, key highlighted formulas - share one highlight treatment: a thin orange border plus a soft orange glow shadow, on a light, mostly transparent surface. Reuse this exact "neon outline" treatment everywhere it applies. Do not invent a new highlight style per section.

Do not restyle the base typography, spacing, or color system per section of the playground. The minimal foundation stays constant throughout; only the additional per-section visual motifs (see "Section Identity and Narrative Progression") change.

---

# Responsive Design

The playground must work well on both desktop and smaller screens.

Do not assume a large desktop viewport.

When designing a visualization:

- allow layouts to reflow
- avoid unnecessary fixed dimensions
- avoid horizontal page overflow
- keep controls usable on touch devices
- make labels readable
- make interactive targets large enough to use comfortably

A visualization that only works at one screen size is incomplete.

---

# Animation

Animations are an important part of the experience.

They must feel smooth and intentional.

Avoid animation simply for decoration.

Animation should explain a transition, relationship, or consequence.

Examples:

- smoothly moving a regression line when coefficients change
- smoothly updating residuals
- showing how a fitted model reacts to changing observations

Prefer continuous visual transitions instead of abruptly recreating the entire visualization.

Avoid expensive rendering work on every frame when it is not necessary.

Use `requestAnimationFrame` where appropriate.

Do not create unnecessary React re-renders during animations.

Avoid repeatedly recalculating values that can be memoized or computed only when inputs change.

Animations must not make the application sluggish.

Respect `prefers-reduced-motion` where practical.

---

# Performance

Performance is part of UX.

The playground should remain responsive even when animations are active.

When modifying or adding functionality, consider:

- unnecessary React re-renders
- expensive calculations inside render functions
- unnecessary DOM nodes
- large images/assets
- repeated allocation of large arrays
- expensive SVG updates
- animation loops
- event listener cleanup
- timers
- memory leaks
- unnecessary dependencies
- bundle size

Do not optimize blindly.

First identify an actual or likely bottleneck, then improve it.

Prefer simple implementations.

For expensive computations:

- compute only when necessary
- memoize when it provides measurable value
- separate simulation/calculation state from visual rendering when useful

For animation:

- keep work per frame small
- avoid updating unrelated components
- stop animation loops when they are not visible or no longer needed

---

# React Architecture

Keep components reasonably small and focused.

Separate:

- educational/model logic
- visualization logic
- UI controls

when this makes the implementation easier to understand.

Do not introduce abstractions purely for architectural purity.

Avoid both extremes:

- one enormous component
- dozens of tiny abstractions that make the code difficult to follow

Prefer code that another developer can understand quickly.

Reuse common visualization/control patterns when genuine repetition appears.

---

# Mathematical Logic

The playground demonstrates mathematical and statistical concepts.

Keep mathematical calculations separate enough from presentation code that they can be inspected and tested.

Prefer small pure functions for calculations such as:

- predictions
- residuals
- error metrics
- regression coefficients
- transformations

Do not duplicate mathematical implementations across components if they represent the same operation.

Numerical behavior should be predictable and robust for unusual user inputs.

---

# Adding New Educational Material

New material must be added iteratively.

Do NOT implement many new demonstrations in one large change.

Use this process:

1. Identify one useful concept.
2. Decide what the user should be able to manipulate.
3. Decide what visual consequence they should observe.
4. Implement the smallest useful version.
5. Make the interaction understandable.
6. Polish the visual design.
7. Optimize animation and rendering.
8. Verify responsive behavior.
9. Only then move to another concept.

A new demonstration is not finished merely because it technically works.

It should also feel polished.

---

# Selecting What to Add

Before adding an interactive concept, ask:

1. Does interaction improve understanding compared with the static article?
2. Is there something meaningful for the user to manipulate?
3. Is the visual response immediate and understandable?
4. Does this concept benefit from experimentation?

If the answer is mostly "no", leave the concept in the article instead.

Not every section of the article needs an interactive equivalent.

---

# Progressive Complexity

Prefer demonstrations that reveal complexity gradually.

Start with a useful default state.

Do not show ten controls immediately if two are enough.

When possible:

`simple initial state → optional deeper exploration`

The first screen should make sense without configuration.

Good defaults are important.

---

# Avoid Feature Creep

Do not add features just because they are technically interesting.

Do not turn the playground into:

- a general statistics calculator
- a full ML framework
- a notebook replacement
- a plotting application
- a complete linear regression course

The playground exists to provide a few memorable interactive explanations.

---

# Existing Content

When improving existing demonstrations, preserve their educational intent unless there is a clear reason to change it.

It is acceptable to:

- simplify controls
- remove unnecessary UI
- improve visual hierarchy
- improve animations
- improve responsive behavior
- improve performance
- refactor implementation
- improve accessibility

It is also acceptable to remove an existing feature if it adds complexity but little educational value.

When considering removal, explain the reasoning before making a large destructive change.

---

# Working Method

For substantial tasks, work in this order:

## 1. Inspect

Understand the relevant implementation first.

## 2. Evaluate

Identify concrete problems in:

- UX
- visual design
- performance
- architecture
- educational clarity

Do not invent problems merely to justify refactoring.

## 3. Plan

Propose a small coherent improvement.

Avoid large rewrites unless necessary.

## 4. Implement

Make the smallest set of changes necessary.

## 5. Validate

Run the application and build it.

Check the actual rendered experience, not only the source code.

## 6. Polish

After functionality works, review:

- spacing
- typography
- alignment
- animation timing
- control placement
- responsive behavior
- unnecessary visual noise

---

# Validation

Before considering a change complete:

- run the frontend locally
- verify there are no console errors
- verify the production build succeeds
- verify existing interactions still work
- verify animations remain smooth
- verify the layout works at different viewport sizes
- check for obvious unnecessary re-renders or animation work
- check that no new large dependency was introduced without a good reason

Use the existing project tooling.

Do not change deployment architecture unless explicitly requested.

The application must continue to work when deployed under:

`/linear-regression/interactive/`

---

# Dependencies

Prefer the existing stack.

Do not introduce a large UI, animation, visualization, or state-management framework merely to solve a small problem.

A new dependency should provide clear value that would otherwise require substantial custom implementation.

Avoid dependency churn.

---

# Refactoring

Refactoring is encouraged when it makes future interactive development easier, but it should serve a concrete purpose.

Good reasons:

- performance problems
- duplicated mathematical logic
- difficult-to-maintain animation code
- components that prevent responsive design
- repeated visualization patterns
- code that makes adding the next demonstration unnecessarily difficult

Bad reason:

> "This architecture is more fashionable."

---

# Priority Order

When trade-offs are necessary, prioritize:

1. Educational clarity
2. Pleasant user experience
3. Visual quality
4. Smooth interaction and performance
5. Simplicity of implementation
6. Quantity of content

Quantity is deliberately last.

---

# Guiding Principle

The playground should feel like a carefully designed interactive companion to the article, not a collection of demos.

When uncertain whether to add more functionality or polish what already exists:

> polish what already exists first.
