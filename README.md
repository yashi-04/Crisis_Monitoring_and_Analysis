# Crisis Monitoring System

A comprehensive system for monitoring and analyzing crisis-related discussions on Reddit. This project helps identify and track mental health distress, substance use, and suicidality-related content across various mental health subreddits.


### Task 1: Social Media Data Extraction & Preprocessing (API Handling & Text Cleaning)

### Deliverables
* A Python script that retrieves and stores filtered social media posts => src/data_extraction.py 
* A cleaned dataset ready for NLP analysis => data/reddit_posts.csv

### Task 2: Sentiment & Crisis Risk Classification (NLP & Text Processing)
### Deliverables
* A script that classifies posts based on sentiment and risk level => src/sentiment_analysis.py
* A table or plot showing the distribution of posts by sentiment and risk category => data/sentiment_by_risk.png, sentiment_risk_distribution.png

### Task 3: Crisis Geolocation & Mapping (Basic Geospatial Analysis & Visualization)

### Deliverables
* A Python script that geocodes posts and generates a heatmap of crisis discussions => src/geolocation.py and data/crisis_heatmap.html
* A visualization of regional distress patterns in the dataset => data/regional_distress_patterns.png


## Features
- Reddit data extraction from mental health subreddits
- Enhanced location extraction from post content
- Sentiment analysis using VADER and TextBlob
- Risk level classification (High, Moderate, Low)
- Location extraction and geocoding
- Interactive heatmap visualization
- Top 5 location analysis

## Project Structure

```
crisis_monitor/
├── data/                  # Directory for storing data files
├── src/                   # Source code
│   ├── data_extraction.py # Reddit data collection
│   ├── sentiment_analysis.py # Sentiment and risk analysis
│   └── geolocation.py    # Location analysis and mapping
└── requirements.txt      # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Create a `.env` file in the project root with your Reddit API credentials:
```
# Reddit API Credentials
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```

## Usage

1. Extract Reddit data:
```bash
python src/data_extraction.py
```

2. Analyze sentiment and risk levels:
```bash
python src/sentiment_analysis.py
```

3. Generate location analysis and heatmap:
```bash
python src/geolocation.py
```

## Output Files

- `data/reddit_posts.csv`: Raw Reddit data with extracted locations
- `data/analyzed_posts.csv`: Posts with sentiment and risk analysis
- `data/location_analyzed_posts.csv`: Posts with geocoded location information
- `data/crisis_heatmap.html`: Interactive heatmap visualization

## Dependencies

- pandas
- numpy
- praw
- nltk
- textblob
- vaderSentiment
- spacy
- folium
- plotly
- geopy
- python-dotenv
- emoji

## Notes

- The system uses rate limiting for Reddit API calls to comply with platform restrictions
- Location extraction uses both NLP and pattern matching to identify locations in posts
- The heatmap visualization weights posts by risk level
- All data is stored locally in CSV format for privacy and compliance
- Monitored subreddits include: depression, anxiety, mentalhealth, suicidewatch, addiction, ptsd, bipolar, schizophrenia, mentalillness, psychology, therapy, counseling 
