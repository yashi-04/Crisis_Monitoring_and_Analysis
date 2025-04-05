import os
import praw
import pandas as pd
from datetime import datetime, timedelta
import re
import emoji
from dotenv import load_dotenv

# API Crendentials are saved in a .env 
load_dotenv()

class RedditExtractor:
    def __init__(self):
        # Reddit API credentials loaded from .env
        self.reddit_client = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        
        # Crisis-related keywords to search for in posts
        self.crisis_keywords = [
            "depressed", "anxiety", "suicidal", "overwhelmed",
            "addiction", "help needed", "crisis", "mental health",
            "therapy", "counseling", "self harm", "hopeless",
            "can't cope", "breaking down", "need support"
        ]
        
        # Mental health related subreddits to search for posts
        self.subreddits = [
            'depression', 'anxiety', 'mentalhealth', 'suicidewatch',
            'addiction', 'ptsd', 'bipolar', 'schizophrenia',
            'mentalillness', 'psychology', 'therapy', 'counseling'
        ]
    
    def clean_text(self, text):
        """Clean text by removing URLs, emojis, and special characters"""
        if not isinstance(text, str):
            return ""
            
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Removing emojis for better text analysis
        # Converting emojis to their text representation and then removing them
        text = emoji.demojize(text)
        text = re.sub(r':[a-zA-Z_]+:', '', text)
        
        # Removing special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text
    
    def extract_location_from_text(self, text):
        # Common location indicators to extract location from post text
        location_indicators = [
            r'in\s+([A-Za-z\s,]+)',  
            r'from\s+([A-Za-z\s,]+)', 
            r'at\s+([A-Za-z\s,]+)',  
            r'located\s+in\s+([A-Za-z\s,]+)',  
            r'based\s+in\s+([A-Za-z\s,]+)',  
            r'area\s+of\s+([A-Za-z\s,]+)',  
            r'near\s+([A-Za-z\s,]+)', 
            r'around\s+([A-Za-z\s,]+)'  
        ]
        
        for pattern in location_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                location = matches[0].strip()
                return location
        
        return None
    
    def extract_reddit_data(self, days_back=7):
        """Extract Reddit posts related to mental health crisis"""
        reddit_data = []
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                for post in subreddit.new(limit=100):
                    # Checking whether the post is within time range
                    post_time = datetime.fromtimestamp(post.created_utc)
                    if post_time < start_time:
                        continue
                    
                    # Combining title and content for keyword matching
                    full_text = f"{post.title} {post.selftext}"
                    
                    if any(keyword in full_text.lower() for keyword in self.crisis_keywords):
                        # Extracting location from post
                        location = self.extract_location_from_text(full_text)
                        
                        reddit_data.append({
                            'platform': 'reddit',
                            'post_id': post.id,
                            'subreddit': subreddit_name,
                            'timestamp': post_time,
                            'title': post.title,
                            'content': post.selftext,
                            'cleaned_content': self.clean_text(full_text),
                            'upvotes': post.score,
                            'comments': post.num_comments,
                            'location': location,
                            'author': post.author.name if post.author else '[deleted]',
                            'url': f"https://reddit.com{post.permalink}"
                        })
            except Exception as e:
                print(f"Error fetching Reddit posts from r/{subreddit_name}: {str(e)}")
        
        return pd.DataFrame(reddit_data)
    
    def save_data(self, df, filename='reddit_posts.csv'):
        os.makedirs('data', exist_ok=True)
        
        #Saving the data to a CSV file
        df.to_csv(f'data/reddit_posts.csv', index=False)
        print(f"Data saved to data/reddit_posts.csv")

def main(): 
    #Main function to execute the data extraction
    extractor = RedditExtractor()
    
    # Extracting Reddit data
    reddit_df = extractor.extract_reddit_data()
    
    # Saving the data
    extractor.save_data(reddit_df)

if __name__ == "__main__":
    main() 
