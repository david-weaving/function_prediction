﻿# Function Predictor

[Try it here!](https://www.functionprediction.com/)

## How it works
This app is a function predictor that takes in 6 (x,y) points. These points are passed through an AI model that determines the function type: polynomial, exponential, sine, or natural log. Then, the points are fitted to the predicted function type through mathematical algorithms. If the function is determined to be polynomial, another AI model will further predict the degree (ranging from 1 to 5). Once fitted, the remaining points are generated, and the function itself is returned.

## How to use it

Using the function predictor is very simple. First, you need 6 (x,y) points. The function predictor only works with exactly 6 points. You can pull these from a data set, an existing function, or even make them up yourself! Ensure that your points are separated by commas.

So, if you have the __x-values__: [1 , 2 , 3 , 4 , 5 , 6] and the __y-values__: [1 , 4 , 9 , 16 , 25 , 36] 

### Your points will plot as:

(1 , 1) , (2 , 4) , (3 , 9) , (4 , 16) , (5 , 25) , (6 , 36)


![Not Fitted Example Points](./images/example%20points%20(not%20fitted).JPG)

### Then the predictor will fit your points to a function:

![Fitted Example Points](./images/example%20points%20(fitted).JPG)

### The predictor will also return the function type and the function itself:

![Prediction](./images/predicted.JPG)


## Interactive Prediction

You can try the interactive prediction [here!](https://www.functionprediction.com/choose)

Instead of typing in 6 x-values and 6 y-values, you can simply click 6 points anywhere on the graph and fit your points!

![Interactive](./images/inter.JPG)


## Predict Myself

You can try predict myself [here!](https://www.functionprediction.com/user_predict)

With 'Predict Myself' you can use as many (and as little) points as you'd like. You can fit those points to any of the four functions available (Polynomial, Sine, Natural Log, Exponential).

![Myself](./images/myse.JPG)
