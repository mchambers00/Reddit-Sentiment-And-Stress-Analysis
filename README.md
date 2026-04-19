# Reddit Stress and Sentiment Analysis

## Overview
This project analyzes Reddit posts from stress-related and support-focused communities to explore patterns in sentiment, engagement, and text characteristics. Using Python, NLP techniques, and data visualization, the analysis focuses on how post content varies across subreddits and how engagement metrics like comments relate to social karma.

## Objectives
- Clean and prepare Reddit post data for analysis  
- Measure sentiment in post text using VADER sentiment analysis  
- Compare engagement and text trends across multiple subreddits  
- Explore relationships between comment volume and social karma  
- Visualize linguistic and engagement patterns  

## Tools & Technologies
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- NLTK (VADER Sentiment Analysis)  
- scikit-learn  
- WordCloud  

## Dataset Features
The dataset includes Reddit post data such as:
- `subreddit`
- `text`
- `social_timestamp`
- `social_karma`
- `social_num_comments`

### Engineered Features
- Word count per post  
- Subreddit ID mapping  
- Time-based features (hour, month, year)  
- Sentiment score (`compound` from VADER)  

## Methodology
1. Loaded and explored Reddit dataset  
2. Removed unnecessary columns to focus analysis  
3. Created new features (word count, timestamps, subreddit IDs)  
4. Applied VADER sentiment analysis to calculate compound sentiment scores  
5. Aggregated sentiment and engagement by subreddit  
6. Built a linear regression model to analyze relationships between comments and karma  
7. Generated visualizations to explore patterns  

## Analysis Performed
- Sentiment scoring of Reddit posts  
- Mean sentiment comparison across subreddits  
- Average word count by subreddit  
- Post frequency distribution  
- Word frequency analysis using tokenization and stopword removal  
- Word cloud generation  
- Linear regression modeling of engagement metrics  

## Key Insights
- Different subreddits exhibit distinct patterns in sentiment and engagement  
- Sentiment analysis provides measurable insights into emotional tone across communities  
- Comment volume shows some relationship with social karma, though the model is relatively simple  
- Text processing highlights common themes in stress-related discussions  

## Visualizations
- Bar charts for subreddit activity and average metrics  
- Scatter plot for regression analysis  
- Word frequency plots  
- Word cloud for common terms  

## Future Improvements
- Apply advanced NLP techniques (lemmatization, embeddings, transformers)  
- Improve regression model with additional features  
- Compare multiple machine learning models  
- Build a reproducible pipeline or notebook version  
- Add interactive dashboards (Tableau or Plotly)  

## How to Run
1. Place `Reddit.csv` in the project directory  
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib nltk scikit-learn wordcloud
