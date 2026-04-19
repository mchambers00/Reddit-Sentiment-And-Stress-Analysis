Reddit Stress and Sentiment Analysis
Overview

This project analyzes Reddit posts from stress-related and support-focused communities to explore patterns in sentiment, engagement, and text characteristics. Using Python, NLP techniques, and data visualization, the analysis focuses on how post content varies across subreddits and how engagement metrics like comments relate to social karma.

Objectives
Clean and prepare Reddit post data for analysis
Measure sentiment in post text using VADER sentiment analysis
Compare engagement and text trends across multiple subreddits
Explore whether post activity, such as comment count, relates to karma
Visualize linguistic and engagement patterns in stress-related communities.
Tools Used
Python
Pandas
NumPy
Matplotlib
NLTK
scikit-learn
WordCloud
Dataset Features Used

The analysis used Reddit post fields including:

subreddit
text
social timestamp
social karma
social number of comments

Additional engineered features included:

word count
subreddit ID mapping
hour, month, and year from timestamps
compound sentiment score
Methodology
Loaded and inspected the Reddit dataset
Removed unnecessary columns and reduced the feature set
Created new analytical features such as word count and timestamp-based variables
Applied VADER sentiment analysis to calculate compound sentiment scores for each post
Grouped and compared sentiment and engagement by subreddit
Built a linear regression model to examine the relationship between number of comments and karma
Visualized subreddit activity, word frequency, karma trends, and text patterns
Key Analysis Performed
Sentiment scoring of Reddit post text
Average sentiment by subreddit
Mean word count by subreddit
Post count distribution across communities
Word frequency analysis using tokenization and stopword removal
Word cloud generation for common post terms
Linear regression using comment count to predict karma
Key Insights
Different support-oriented subreddits showed measurable variation in text volume and engagement
Sentiment analysis provided a way to quantify emotional tone across stress-related communities
Comment volume showed some predictive relationship with social karma, though the model was simple and could be improved with additional features
Text preprocessing and visualization helped identify common themes across posts
Possible Improvements
Add stronger text preprocessing, such as punctuation cleanup and lemmatization
Use more predictive features for engagement modeling
Compare multiple machine learning models instead of only linear regression
Improve sentiment analysis by comparing VADER with transformer-based NLP methods
Refine visualizations and package the notebook/script into a more reproducible pipeline
How to Run
Place Reddit.csv in the project directory
Install dependencies:
pandas
numpy
matplotlib
nltk
scikit-learn
wordcloud
Run the Python script
Download required NLTK resources when prompted:
punkt
vader_lexicon
stopwords
