import pandas as pd
import spacy
import folium
from folium.plugins import HeatMap, MarkerCluster
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

class LocationAnalyzer:
    def __init__(self):
        
        self.nlp = spacy.load("en_core_web_sm")
        
       
        self.geolocator = Nominatim(user_agent="crisis_monitor")
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1)
        
        
        self.us_cities = {
            'new york', 'los angeles', 'chicago', 'houston', 'phoenix',
            'philadelphia', 'san antonio', 'san diego', 'dallas', 'san jose',
            'austin', 'jacksonville', 'fort worth', 'columbus', 'charlotte',
            'san francisco', 'indianapolis', 'seattle', 'denver', 'washington',
            'boston', 'nashville', 'detroit', 'portland', 'memphis',
            'oklahoma city', 'las vegas', 'louisville', 'baltimore', 'milwaukee',
            'albuquerque', 'tucson', 'fresno', 'sacramento', 'mesa',
            'kansas city', 'atlanta', 'miami', 'omaha', 'raleigh',
            'minneapolis', 'cleveland', 'wichita', 'arlington', 'new orleans'
        }
        
        self.us_states = {
            'alabama', 'alaska', 'arizona', 'arkansas', 'california',
            'colorado', 'connecticut', 'delaware', 'florida', 'georgia',
            'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas',
            'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts',
            'michigan', 'minnesota', 'mississippi', 'missouri', 'montana',
            'nebraska', 'nevada', 'new hampshire', 'new jersey', 'new mexico',
            'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma',
            'oregon', 'pennsylvania', 'rhode island', 'south carolina',
            'south dakota', 'tennessee', 'texas', 'utah', 'vermont',
            'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming'
        }
        
       
        self.state_abbreviations = {
            'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 
            'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 
            'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID', 
            'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS', 
            'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD', 
            'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS', 
            'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV', 
            'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 
            'new york': 'NY', 'north carolina': 'NC', 'north dakota': 'ND', 
            'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR', 'pennsylvania': 'PA', 
            'rhode island': 'RI', 'south carolina': 'SC', 'south dakota': 'SD', 
            'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT', 'vermont': 'VT', 
            'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV', 
            'wisconsin': 'WI', 'wyoming': 'WY'
        }
        
        #location patterns
        self.location_patterns = [
            r'in\s+([A-Za-z\s]+),\s*([A-Z]{2})',  
            r'from\s+([A-Za-z\s]+),\s*([A-Z]{2})',  
            r'at\s+([A-Za-z\s]+),\s*([A-Z]{2})',  
            r'near\s+([A-Za-z\s]+),\s*([A-Z]{2})',  
            r'around\s+([A-Za-z\s]+),\s*([A-Z]{2})', 
            r'based\s+in\s+([A-Za-z\s]+),\s*([A-Z]{2})',  
            r'located\s+in\s+([A-Za-z\s]+),\s*([A-Z]{2})',  
            r'stuck\s+in\s+([A-Za-z\s]+),\s*([A-Z]{2})',  
            r'stranded\s+in\s+([A-Za-z\s]+),\s*([A-Z]{2})',  
            r'need\s+help\s+in\s+([A-Za-z\s]+),\s*([A-Z]{2})',  
            r'crisis\s+in\s+([A-Za-z\s]+),\s*([A-Z]{2})',  
            r'emergency\s+in\s+([A-Za-z\s]+),\s*([A-Z]{2})',  
            r'([A-Za-z\s]+),\s*([A-Z]{2})',  
            r'([A-Za-z\s]+)\s+([A-Z]{2})', 
            r'([A-Z]{2})\s+area', 
            r'([A-Z]{2})\s+region',  
            r'([A-Z]{2})\s+state', 
        ]
        
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.location_patterns]
    
    def extract_location(self, text):
        """Extract location from text using NLP and pattern matching"""
        if not isinstance(text, str):
            return None
            
        
        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                if len(match.groups()) == 2:
                    
                    city, state = match.groups()
                    
                    city = city.strip().lower()
                   
                    state = state.upper()
                    if state in self.state_abbreviations.values():
                        return f"{city}, {state}"
                elif len(match.groups()) == 1:
                    
                    state = match.group(1).upper()
                    if state in self.state_abbreviations.values():
                        return state
        
        
        doc = self.nlp(text)
        locations = []
        
        
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:
                locations.append(ent.text.lower())
        
        
        words = text.lower().split()
        for i in range(len(words)):
            if words[i] in self.us_cities:
                locations.append(words[i])
            elif words[i] in self.us_states:
                locations.append(words[i])
        
       
        if locations:
            return locations[0]
        
        
        state_abbr_pattern = re.compile(r'\b([A-Z]{2})\b')
        state_matches = state_abbr_pattern.findall(text)
        for state in state_matches:
            if state in self.state_abbreviations.values():
                return state
        
        return None
    
    def geocode_location(self, location):
        """Convert location name to coordinates"""
        if not location:
            return None
            
        try:
            
            if location.lower() in self.us_cities or location.lower() in self.us_states:
                location = f"{location}, USA"
            
            location_data = self.geocode(location)
            if location_data:
                return {
                    'latitude': location_data.latitude,
                    'longitude': location_data.longitude,
                    'address': location_data.address
                }
        except Exception as e:
            print(f"Error geocoding {location}: {str(e)}")
        
        return None
    
    def analyze_locations(self, df):
        """Analyze locations in the dataset"""
        
        df['location'] = df['cleaned_content'].apply(self.extract_location)
        
        
        df['coordinates'] = df['location'].apply(self.geocode_location)
        
        
        df['latitude'] = df['coordinates'].apply(lambda x: x['latitude'] if x else None)
        df['longitude'] = df['coordinates'].apply(lambda x: x['longitude'] if x else None)
        
        
        df['state'] = df['location'].apply(self.extract_state)
        
        return df
    
    def extract_state(self, location):
        """Extract state from location string"""
        if not location:
            return None
            
        location_lower = location.lower()
        
        
        if location_lower in self.us_states:
            return self.state_abbreviations.get(location_lower)
        
       
        for state, abbr in self.state_abbreviations.items():
            if state in location_lower:
                return abbr
        
        return None
    
    def create_heatmap(self, df, output_file='crisis_heatmap.html'):
        """Create a heatmap of crisis-related posts"""
        
        os.makedirs('data', exist_ok=True)
        
        
        valid_posts = df[df['latitude'].notna() & df['longitude'].notna()]
        
        
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        
        
        heatmap_data = []
        for _, row in valid_posts.iterrows():
            
            weight = 3 if row['risk_level'] == 'High' else 2 if row['risk_level'] == 'Moderate' else 1
            heatmap_data.append([row['latitude'], row['longitude'], weight])
        
        
        HeatMap(heatmap_data).add_to(m)
        
       
        marker_cluster = MarkerCluster().add_to(m)
        
        #Adding high-risk posts to the heatmap
        for _, row in valid_posts[valid_posts['risk_level'] == 'High'].iterrows():
            popup_text = f"""
            <b>Risk Level:</b> {row['risk_level']}<br>
            <b>Sentiment:</b> {row['sentiment']}<br>
            <b>Location:</b> {row['location']}<br>
            <b>Content:</b> {row['content'][:100]}...
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(marker_cluster)
        
        
        m.save(f'data/crisis_heatmap.html')
        print(f"\nHeatmap saved to data/crisis_heatmap")
    
    def create_regional_analysis(self, df):
        """Create visualizations of regional distress patterns"""
        
        os.makedirs('data', exist_ok=True)
        
        
        state_df = df[df['state'].notna()]
        
        if len(state_df) == 0:
            print("No state data available for regional analysis")
            return
        
        
        state_df['state_full_name'] = state_df['state'].apply(self.get_full_state_name)
        
        
        state_counts = state_df['state_full_name'].value_counts()
        
        
        state_risk = pd.crosstab(state_df['state_full_name'], state_df['risk_level'])
        
        
        state_sentiment = pd.crosstab(state_df['state_full_name'], state_df['sentiment'])
        
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        

        state_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Number of Crisis Posts by State')
        axes[0, 0].set_xlabel('State')
        axes[0, 0].set_ylabel('Number of Posts')
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        
        for i, v in enumerate(state_counts.values):
            axes[0, 0].text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        #Bar plot of risk levels by state
        state_risk.plot(kind='bar', stacked=True, ax=axes[0, 1], colormap='viridis')
        axes[0, 1].set_title('Risk Levels by State')
        axes[0, 1].set_xlabel('State')
        axes[0, 1].set_ylabel('Number of Posts')
        axes[0, 1].legend(title='Risk Level')
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        #Heatmap of risk levels by state
        sns.heatmap(state_risk, annot=True, fmt='d', cmap='YlGnBu', ax=axes[1, 0])
        axes[1, 0].set_title('Risk Levels by State (Heatmap)')
        
        #Top 5 states with crisis posts
        top_states = state_counts.head(5)
        axes[1, 1].pie(top_states.values, labels=top_states.index, autopct='%1.1f%%', 
                      startangle=90, colors=sns.color_palette('viridis'))
        axes[1, 1].set_title('Top 5 States with Crisis Posts')
        
        #Saving the regional distress patterns visualization to a PNG file
        plt.tight_layout()
        plt.savefig('data/regional_distress_patterns.png', dpi=300)
        print("\nRegional distress patterns visualization saved to data/regional_distress_patterns.png")
        
        
        return fig
    
    def get_full_state_name(self, state_abbr):
        """Convert state abbreviation to full state name"""
        if not state_abbr:
            return None
            
        
        for full_name, abbr in self.state_abbreviations.items():
            if abbr == state_abbr:
                return full_name.title()
        
        return state_abbr
    
    
    def get_top_locations(self, df, n=5):
        """Top locations with highest crisis discussions"""
        location_counts = df['location'].value_counts()
        top_locations = location_counts.head(n)
        
        print("\nTop 5 Locations with Highest Crisis Discussions:")
        for location, count in top_locations.items():
            print(f"{location}: {count} posts")
        
        return top_locations
    
    def save_location_data(self, df, filename='location_analyzed_posts.csv'):
        """Saving the location-analyzed data to a CSV file"""
       
        os.makedirs('data', exist_ok=True)
        
        df.to_csv(f'data/{filename}', index=False)
        print(f"\nLocation-analyzed data saved to data/{filename}")

def main():
    #Reading the analyzed data
    df = pd.read_csv('data/analyzed_posts.csv')
    
    #Initializing the analyzer      
    analyzer = LocationAnalyzer()
    
    #Analyzing locations in the dataset
    location_df = analyzer.analyze_locations(df)
    
    #Analyzing locations in the dataset     
    analyzer.create_heatmap(location_df)
    
    #Creating a heatmap of crisis-related posts
    analyzer.create_regional_analysis(location_df)
    
    #Top locations with highest crisis discussions
    top_locations = analyzer.get_top_locations(location_df)
    
   
    analyzer.save_location_data(location_df)

if __name__ == "__main__":
    main() 