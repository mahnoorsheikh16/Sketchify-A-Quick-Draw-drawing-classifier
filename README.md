# Sketchify – A Quick, Draw! drawing classifier
Teaching machines to recognize freehand sketches is both a scientific challenge and a playful application of modern AI. One such game is Google Creative Lab’s [Quick, Draw!](https://quickdraw.withgoogle.com/), which is developed to train a neural network to recognize doodling. This serves as the inspiration for this project.

The objective is to develop machine learning models that can classify hand-drawn digital drawings or doodles by identifying the object or category they represent. For this purpose, three Bayesian classifiers, a collection of classical models (Logistic Regression, Support Vector Machine, K-NN, XGBoost), and a Recurrent Neural Network (RNN) have been trained on varying sample sizes. The following sections outline the preprocessing, feature engineering and model evaluation steps, providing insights into each model’s performance.

*This is a course project for CSE 802 Pattern Recognition and Analysis at MSU.

## Table of Contents:
1. [Dataset](#dataset)

## Dataset:
The [Quick Draw Dataset](https://quickdraw.withgoogle.com/data) is a collection of 50 million drawings across 345 categories, contributed by players of the game [Quick, Draw!](https://quickdraw.withgoogle.com/). It has been compiled by Google Creative Lab to help developers train new neural networks, help researchers see patterns in how people around the world draw, and help artists create things we haven’t begun to think of.

The [simplified dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified;tab=objects?invt=AbvrVA&prefix=&forceOnObjectsSortingFiltering=false) consists of drawings recorded as timestamped vectors, tagged with metadata. It includes the following features: 

1. `word`: category of the drawing
   
2. `countrycode`: player's country
   
3. `timestamp`
   
4. `recognized`: indicating whether the model successfully classified the drawing
   
5. `key_id`
    
6. `drawing array`: contains x and y pixel coordinates and t, which represents the time in milliseconds since the first point.

The [Quick Draw dataset](https://quickdraw.withgoogle.com/data) contains 50 million drawings contributed by over 15 million players across 345 categories. For this project, 1000 random drawings from 10 categories each have been sampled, namely ‘apple’, ‘baseball’, ‘bridge’, ‘circle’, ‘cow’, ‘flower’, ‘moustache’, ‘speedboat’, ‘square’, and ‘yoga’.

Categories ‘apple, baseball, circle, flower, square’ display low intraclass and moderate interclass variations; whereas ‘bridge, cow, moustache, speedboat, yoga’ exhibit high intraclass and interclass variations.

Since the raw dataset is so vast, a preprocessed version has been used. The simplified drawing files constitute a simplified drawing vector with:

• drawing aligned to the top-left corner to have minimum values of 0

• uniform scaling into a 256x256 region (to have a maximum value of 255)

• resampled strokes with 1 pixel spacing

• simplified strokes using the Ramer–Douglas–Peucker algorithm with an epsilon value of 2.0

• timing information removed

It includes the following features: `word` (category of the drawing), `countrycode` (player's country), `timestamp` (time the drawing was created), `recognized` (whether the drawing was classified by the model), `key_id` (unique identifier across drawings), and `drawing array` (JSON array representing the vector drawing with 𝑥 and 𝑦 pixel coordinates).

For illustration purposes, the drawing array is of the form: 

`[

[ // First stroke

[x0, x1, x2, x3, ...],

[y0, y1, y2, y3, ...]

],`
