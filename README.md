# SWYM CORPORATION

<br />

## BUSINESS UNDERSTANDING
Swym is a company which aims to help E-Commerce brands craft more seamless and personalized experiences for their shoppers. They are built on the Shopify platform for online stores, and add functionality to those sites in the form of applications such as Wishlist+ and Watchlist. The main goal of this project was to use machine learning algorithms to accurately model customer journeys based on available user context, and to leverage that predictive power to further optimize the online shopping experience for individual users.

## DATA UNDERSTANDING
Swym's extensive data was made accessible to me for querying via Azure SQL. I worked primarily with two large datasets:

* Sessions - A session begins when a user accesses a Shopify online store, and ends once the user is inactive for 30 minutes or more. Each row in this table represents one user action within the given session. This can also be thought of as any change in the URL.

* Devices - Provides additional metadata on the device used to access a Shopify site.

There are thousands of Shopify stores which make use of Swym's products, but I limited the scope of my capstone project to just a few select medium-sized providers.

## DATA PREPARATION
The bulk of my work on this project went into cleaning and converting the available data into a usable format, as well as into engineering new features that may potentially be informative in identifying a customer's next action within the current session. Variables derived include:

* Seasonality indicators for day of week and hour of day
* URL of site customer was referred from
* Device type and operating system
* Elapsed time since beginning of session and since last user action
* Prior actions taken by user in current session
* Access history for the user on each Shopify site
* Tf-idf vectors for page title and product category

I developed a comprehensive Python class named "Swymify" to accomplish all of the above. The class also includes various methods for fitting and evaluating various machine learning models.

## MODELING
Models were trained on data through the full month of February 2017 and tested on data from the same providers for the first week of March 2017. Prediction accuracy was the preferred measure of success. I evaluated three different classification techniques for predicting a Swym user's next action: Random Forest, Gradient Boosting, and Support Vector Machine. I tuned the hyperparameters for each of these three models via Grid Search, and compared them using 5-fold cross validation on the training data. The accuracies are as follows:

## EVALUATION
Accuracy of predictions within test dataset. Will probably cross-validate.

## DEPLOYMENT
Academic Paper detailing results and conclusions, and hopefully a web app where you can input user actions within a given session and see what the next predicted action is. Probability of eventual purchase follows directly and should be output as well.

## NEXT STEPS
Webscraping additional fields from actual page urls (i.e. product descriptions, images)
