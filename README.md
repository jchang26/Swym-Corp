# swym-corp

<br />

## BUSINESS UNDERSTANDING
Swym is a company which aims to help ecommerce brands craft more seamless experiences for their shoppers, personalized around the shoppers' journey. The primary goal for this project is to use predictive algorithms to model online Shopify store customer journeys based on the available user context and leverage those predictions to optimize the experience for shoppers.

## DATA UNDERSTANDING
The main datasets made available to me on Azure SQL are:

* Sessions - Each row represents one user action within a given session, which changes the url. A session is definited as all actions an individual makes within a 30 minute window on a Shopify site.

* Devices - Device metadata

* Users - User Metadata from past transactions

## DATA PREPARATION
Build model pipeline based on small subset of data. Final model built on large scope full dataset, once access to Azure SQL databases is obtained.

## MODELING
Would like to build a model which predicts a user's next action within a given session, given his metadata and the actions he's already made within that session. Will likely be a Markov Chain of some sort. Initially restrict to default actions, but will probably end up adding other possible user actions (i.e. viewing other actions, switching devices, etc.)

## EVALUATION
Accuracy of predictions within test dataset. Will probably cross-validate.

## DEPLOYMENT
Academic Paper detailing results and conclusions, and hopefully a web app where you can input user actions within a given session and see what the next predicted action is. Probability of eventual purchase follows directly and should be output as well.

## POSSIBLE NEXT STEPS
Webscraping additional fields from actual page urls (i.e. product descriptions, images)
