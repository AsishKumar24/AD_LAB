from fastapi import FastAPI, HTTPException, Query
from fastapi.testclient import TestClient  # Import TestClient here
from pydantic import BaseModel
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from googleapiclient.discovery import build
import requests
from fastapi.staticfiles import StaticFiles
# Initialize FastAPI
app = FastAPI()

# Download NLTK resources (only once)
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Option 2: Hard-code your API key here (for testing only; remove in production)
YOUTUBE_API_KEY = "AIzaSyDJFIHFyO0PRzUxhwaZxD0CZ03HKBaVHY0"

# Function to analyze sentiment
def analyze_sentiment(text: str):
    """Returns sentiment (positive, negative, neutral) and scores."""
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    sentiment = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    return {"sentiment": sentiment, "scores": scores}

# Route: Test Sentiment for Custom Text
@app.get("/sentiment")
def get_sentiment(text: str = Query(..., description="Text to analyze for sentiment")):
    """Analyze sentiment of a given text."""
    return analyze_sentiment(text)

# Pydantic Model for YouTube Video Request
class VideoRequest(BaseModel):
    video_id: str

# Route: Analyze YouTube Video Comments
@app.post("/video_sentiment")
def video_sentiment(request: VideoRequest):
    """Fetch YouTube comments and analyze sentiment."""
    if not YOUTUBE_API_KEY:
        raise HTTPException(status_code=500, detail="YouTube API key not configured. Either set the YOUTUBE_API_KEY environment variable or hard-code the key in the code.")

    try:
        # Initialize YouTube API client
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

        # Fetch comments from video
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=request.video_id,
            textFormat="plainText",
            maxResults=50  # Adjust or implement pagination as needed
        ).execute()

        comments = [
            item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for item in response.get("items", [])
        ]

        # Analyze sentiment for each comment
        results = [{"comment": comment, **analyze_sentiment(comment)} for comment in comments]

        return {"video_id": request.video_id, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Test Client for Internal Testing
client = TestClient(app)  # Now TestClient is correctly imported

# Route: Internal Test API (Calls video_sentiment API)
@app.get("/test_video_sentiment")
def test_video_sentiment(video_id: str):
    """Calls the /video_sentiment API internally."""
    response = client.post("/video_sentiment", json={"video_id": video_id})
    return response.json()

# Route: External Call Using Requests
@app.get("/call_video_sentiment")
def call_video_sentiment(video_id: str):
    """Calls the /video_sentiment API using requests."""
    url = "http://127.0.0.1:8000/video_sentiment"
    response = requests.post(url, json={"video_id": video_id})
    return response.json()
