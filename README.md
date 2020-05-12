# TDI_Stock_Price_Prediction

A proposed idea for The Data Incubator Capstone project.


Predicting stock price is an essential prerequisite towards building an AI driven robo adviser. It would be of interest, particularly, to people who are trying to save for retirement. Instead of keeping the money lying around in a low interest savings account, investment in stock market is a judicious choice. When building an AI for stock market, one must take into account historical data, public sentiment about the stock, randomness of the market, and also local inefficiency of the market, i.e., some people have more information about the market than the others. However, due to practical issues, it's not always possible to factor in all these effects into a machine learning model. My goal is to incorporate as much of these effects as possible.

Collecting data: To collect data, I plan to use publicly available data in various websites. As part of these, I will write a web crawler to collect data.

Feature Engineering: 7 and 21 days moving average, exponential moving average, momentum, Bollinger bands, MACD, analysis of the 10-K and 10-Q reports etc.

Public sentiment: I will incorporate twitter feed of notable public figures into the model using Natural Language Processing.

Stacked autoencoders: A derivative of variational autoencoder will be used to extract high level feature from the intermediate layers.

Feature importance: To identify the importance of different features, I will use an integration of XGBoost and SHAP package.

Neural network for stock prediction: I will use LSTM and CNN to build the neural network for the stock price prediction.

Reinforcement learning: FInally, I will use reinforcement learning to gradually imporve the model performance.

Model deployment: To make the model publicly available, I will use a Django Framework. 
