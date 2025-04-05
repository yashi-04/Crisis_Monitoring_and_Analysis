import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

class CrisisAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        
        self.high_risk_patterns = [
            r'suicid[ea]', r'kill\s+myself', r'end\s+it\s+all',
            r'don\'t\s+want\s+to\s+live', r'can\'t\s+take\s+it\s+anymore',
            r'goodbye\s+world', r'final\s+goodbye', r'last\s+post',
            r'planning\s+to\s+die', r'going\s+to\s+die'
        ]
        
        
        self.moderate_risk_patterns = [
            r'help\s+needed', r'can\'t\s+cope', r'feeling\s+lost',
            r'need\s+support', r'struggling', r'overwhelmed',
            r'can\'t\s+sleep', r'panic\s+attack', r'anxiety',
            r'depression', r'hopeless'
        ]
        
        
        self.high_risk_regex = re.compile('|'.join(self.high_risk_patterns), re.IGNORECASE)
        self.moderate_risk_regex = re.compile('|'.join(self.moderate_risk_patterns), re.IGNORECASE)
    
    def get_vader_sentiment(self, text):
        """Sentiment scores using VADER"""
        scores = self.vader.polarity_scores(text)
        return scores['compound']
    
    def get_textblob_sentiment(self, text):
        """Sentiment scores using TextBlob"""
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def classify_risk_level(self, text):
        """Classify the risk level of a post"""
        if self.high_risk_regex.search(text):
            return 'High'
        elif self.moderate_risk_regex.search(text):
            return 'Moderate'
        else:
            return 'Low'
    
    def analyze_posts(self, df):
        """Analyze posts for sentiment and risk level"""
       
        df['vader_sentiment'] = df['cleaned_content'].apply(self.get_vader_sentiment)
        df['textblob_sentiment'] = df['cleaned_content'].apply(self.get_textblob_sentiment)
        
        
        df['risk_level'] = df['cleaned_content'].apply(self.classify_risk_level)
        
        
        df['sentiment'] = df['vader_sentiment'].apply(lambda x: 
            'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral')
        
        return df
    
    def get_risk_statistics(self, df):
        """Statistics about risk levels and sentiment"""
        risk_stats = df.groupby('risk_level').size()
        sentiment_stats = df.groupby('sentiment').size()
        
        print("\nRisk Level Distribution:")
        print(risk_stats)
        print("\nSentiment Distribution:")
        print(sentiment_stats)
        
        return risk_stats, sentiment_stats
    
    def create_distribution_table(self, df):
        """Table showing the distribution of posts by sentiment and risk category"""
        
        distribution_table = pd.crosstab(df['sentiment'], df['risk_level'])
        
       
        distribution_table['Total'] = distribution_table.sum(axis=1)
        distribution_table.loc['Total'] = distribution_table.sum()
        
        
        total_posts = len(df)
        percentage_table = (distribution_table / total_posts * 100).round(2)
        
        
        percentage_table = percentage_table.applymap(lambda x: f"{x}%")
        
        print("\nDistribution of Posts by Sentiment and Risk Category:")
        print(distribution_table)
        print("\nPercentage Distribution:")
        print(percentage_table)
        
        return distribution_table, percentage_table
    
    def create_distribution_plots(self, df):
        """Plots showing the distribution of posts by sentiment and risk category"""
        
        os.makedirs('data', exist_ok=True)
        
        
        sns.set(style="whitegrid")
        
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        #Bar plot of risk level distribution
        risk_counts = df['risk_level'].value_counts()
        sns.barplot(x=risk_counts.index, y=risk_counts.values, ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('Distribution of Risk Levels')
        axes[0, 0].set_xlabel('Risk Level')
        axes[0, 0].set_ylabel('Number of Posts')
        
        
        for i, v in enumerate(risk_counts.values):
            axes[0, 0].text(i, v + 0.5, str(v), ha='center')
        
        #Bar plot of sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=axes[0, 1], palette='viridis')
        axes[0, 1].set_title('Distribution of Sentiments')
        axes[0, 1].set_xlabel('Sentiment')
        axes[0, 1].set_ylabel('Number of Posts')
        
        
        for i, v in enumerate(sentiment_counts.values):
            axes[0, 1].text(i, v + 0.5, str(v), ha='center')
        
        #Heatmap of sentiment vs risk level
        cross_tab = pd.crosstab(df['sentiment'], df['risk_level'])
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu', ax=axes[1, 0])
        axes[1, 0].set_title('Sentiment vs Risk Level')
        
        #Pie chart of risk level distribution
        axes[1, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                      startangle=90, colors=sns.color_palette('viridis'))
        axes[1, 1].set_title('Risk Level Distribution')
        
        #Stacked bar chart of sentiment within each risk level
        plt.tight_layout()
        plt.savefig('data/sentiment_risk_distribution.png', dpi=300)
        print("\nDistribution plots saved to data/sentiment_risk_distribution.png")
        
        
        plt.figure(figsize=(10, 6))
        risk_sentiment = pd.crosstab(df['risk_level'], df['sentiment'])
        risk_sentiment.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('Sentiment Distribution Within Risk Levels')
        plt.xlabel('Risk Level')
        plt.ylabel('Number of Posts')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        plt.savefig('data/sentiment_by_risk.png', dpi=300)
        print("Sentiment by risk level plot saved to data/sentiment_by_risk.png")
        
        return fig
    
    def save_analyzed_data(self, df, filename='analyzed_posts.csv'):
        #Saving the analyzed data to a CSV file
        
        os.makedirs('data', exist_ok=True)
        
        df.to_csv(f'data/{filename}', index=False)
        print(f"\nAnalyzed data saved to data/{filename}")

def main():
    #Main function to execute the sentiment analysis
    df = pd.read_csv('data/reddit_posts.csv')
    
    #Read the data
    analyzer = CrisisAnalyzer()
    
    #Analyze the posts
    analyzed_df = analyzer.analyze_posts(df)
    
    #Statistics about risk levels and sentiment
    risk_stats, sentiment_stats = analyzer.get_risk_statistics(analyzed_df)
    
    #Distribution table and percentage table of sentiment and risk level
    distribution_table, percentage_table = analyzer.create_distribution_table(analyzed_df)
    
    #Plots showing the distribution of posts by sentiment and risk category
    analyzer.create_distribution_plots(analyzed_df)
    
    #Save the analyzed data to a CSV file
    analyzer.save_analyzed_data(analyzed_df)

if __name__ == "__main__":
    main() 