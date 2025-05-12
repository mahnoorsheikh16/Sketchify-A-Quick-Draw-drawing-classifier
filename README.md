# Sketchify – A Quick, Draw! drawing classifier
Teaching machines to recognize freehand sketches is both a scientific challenge and a playful application of modern AI. One such game is Google Creative Lab’s [Quick, Draw!](https://quickdraw.withgoogle.com/), which is developed to train a neural network to recognize doodling. This serves as the inspiration for this project.

The objective is to develop machine learning models that can classify hand-drawn digital drawings or doodles by identifying the object or category they represent. For this purpose, three Bayesian classifiers, a collection of classical models (Logistic Regression, Support Vector Machine, K-NN, XGBoost), and a Recurrent Neural Network (RNN) have been trained on varying sample sizes. The following sections outline the preprocessing, feature engineering and model evaluation steps, providing insights into each model’s performance.

*This is a course project for CSE 802 Pattern Recognition and Analysis at MSU.

## Table of Contents:
1. [Dataset](#dataset)
2. [Initial Data Analysis (IDA)](#initial-data-analysis-ida)

## Dataset:
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

`[`

`[ // First stroke`

`[x0, x1, x2, x3, ...],`

`[y0, y1, y2, y3, ...]`

`],`

`[ // Second stroke`

`[x0, x1, x2, x3, ...],`

`[y0, y1, y2, y3, ...]`

`],`

`... // Additional strokes`

`]`

## Initial Data Analysis (IDA):

**I. Data Cleaning**

Since the dataset only contains individual stroke coordinates for each drawing, features need to be manually extracted. To prepare for this, the drawing array is further simplified.

• Each ndjson line (row) is parsed to extract the class label and the drawing array.

• The stroke data is converted into an array of shape (𝑡𝑜𝑡𝑎𝑙 𝑝𝑜𝑖𝑛𝑡𝑠,3), where columns 1 and 2 store the 𝑥 and 𝑦 pixel coordinates, and column 3 acts as a marker for the end of a stroke (0 by default and 1 at the final point of each stroke).

• 𝑥 and 𝑦 coordinates are scaled [0,1] range, so all doodles share the same spatial range. This avoids introducing bias for drawings with coordinates spanning a larger range and ensures each drawing is assigned an equal weight.

• Deltas are computed for the pixel coordinates, so column 1 now stores the difference between consecutive points for 𝑥 coordinates and column 2 stores the difference between consecutive points for 𝑦 coordinates.

• Redundant first row is dropped since no deltas exist for the first point and the final array has shape (𝑡𝑜𝑡𝑎𝑙 𝑝𝑜𝑖𝑛𝑡𝑠−1,3).

These steps are implemented on each sample and the resulting cleaned dataset has the shape (10000, 2) with 10 classes.

For illustration purposes, the preprocessed drawing array is of the form:

`[`

`[dx0, dy0, m0],`

`[dx1, dy1, m1],`

`... // Additional strokes`

`]`

**II. Feature Engineering**

59 features are extracted from the preprocessed drawing array. These encompass:

Stroke Movement features: `mean_dx, mean_dy, std_dx, std_dy, max_dx, max_dy, min_dx, min_dy, num_strokes, total_points, avg_stroke_len, trajectory_len, longest_stroke, shortest_stroke, ratio_longest_shortest, var_stroke_lengths, avg_jump_distance, std_jump_distance`
*‘dx’ and ‘dy’ are the deltas computed earlier.
*‘jump_distance’ is the spatial distance between stroke-end and next stroke-start.

Statistical features: skew_dx, skew_dy, kurtosis_dx, kurtosis_dy, q25_dx, q75_dx, q25_dy, q75_dy
Geometric/Spatial features: bbox_width, bbox_height, bbox_area, bbox_perimeter, bbox_diagonal, aspect_ratio, centroid_x, centroid_y, start_to_centroid, end_to_centroid, avg_distance_to_centroid, std_distance_to_centroid
*‘bbox’ refers to the bounding box for an image.
Convex Hull features: hull_area, hull_perimeter, solidity
*convex hull is the smallest polygon that encloses all the stroke points.
*‘solidity’ refers to the ratio of the drawing’s filled pixel area to the total hull_area.
Angular/Curvature features: total_angle_change, mean_segment_angle, std_segment_angle, max_angle_change, min_angle_change, avg_curvature, max_curvature, std_curvature
*‘segment_angle’ measures the angle between two consecutive points.
Other features: dominant_frequency, hu_1, hu_2, hu_3, hu_4, hu_5, hu_6, hu_7, fractal_dimension, straightness
*‘dominant_frequency’ captures the dominant back-and-forth pattern/motion repeated in the overall drawing.
*Hue Moments summarize the overall form of the drawing that don’t change if you move, resize, or rotate it, making them invariant features. Each Hu moment represents different weighted sums of pixel positions that, when combined, represent the drawing’s global outline.
*‘fractal_dimension’ measures how detailed/complex/twisty the doodle is by counting the number of boxes of different sizes needed to cover all the strokes.
*‘straightness’ represents if strokes were straighter or wandering.
Finally, the dataset results in shape (10000, 60) with 10 classes. All features have numerical data types and have no missing values.


