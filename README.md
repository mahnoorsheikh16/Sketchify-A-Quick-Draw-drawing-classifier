# Sketchify – A Quick, Draw! drawing classifier
Teaching machines to recognize freehand sketches is both a scientific challenge and a playful application of modern AI. One such game is Google Creative Lab’s Quick, Draw!, which is developed to train a neural network to recognize doodling. This serves as the inspiration for this project.

The objective is to develop machine learning models that can classify hand-drawn digital drawings or doodles by identifying the object or category they represent. For this purpose, three Bayesian classifiers, a collection of classical models (Logistic Regression, Support Vector Machine, K-NN, XGBoost), and a Recurrent Neural Network (RNN) have been trained on varying sample sizes. The following sections outline the preprocessing, feature engineering and model evaluation steps, providing insights into each model’s performance.

*This is a course project for CSE 802 Pattern Recognition and Analysis at MSU.

## Dataset:
The [Quick Draw Dataset](https://quickdraw.withgoogle.com/data) is a collection of 50 million drawings across 345 categories, contributed by players of the game [Quick, Draw!](https://quickdraw.withgoogle.com/). It has been compiled by Google Creative Lab to help developers train new neural networks, help researchers see patterns in how people around the world draw, and help artists create things we haven’t begun to think of.

The [simplified dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified;tab=objects?invt=AbvrVA&prefix=&forceOnObjectsSortingFiltering=false) consists of drawings recorded as timestamped vectors, tagged with metadata. It includes the following features: 

1. `word`: category of the drawing
   
2. `countrycode`: player's country
   
3. `timestamp`
   
4. `recognized`: indicating whether the model successfully classified the drawing
   
5. `key_id`
    
6. `drawing array`: contains x and y pixel coordinates and t, which represents the time in milliseconds since the first point.
