import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer  # VADER Sentiment
import re
import nltk
# Ensure that necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')  # Download the VADER sentiment lexicon

# Function to summarize text
def summarize_text(text, max_length=50000):
    # Simple summary by returning the first `max_length` characters (no model)
    return text[:max_length]

# Function to extract keywords without scikit-learn
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]

    # Filter stop words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]

    # Get word frequencies using FreqDist
    fdist = FreqDist(filtered_words)
    
    # Get the top 5 most frequent words
    top_keywords = [word for word, _ in fdist.most_common(5)]

    return top_keywords

# Function to perform topic modeling without scikit-learn (using word frequencies)
def topic_modeling(text):
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum()]
    
    # Frequency distribution
    fdist = FreqDist(filtered_words)

    # Get the top 5 topics based on word frequencies (this is a simple method, not LDA)
    top_topics = [word for word, _ in fdist.most_common(5)]
    
    return top_topics

# Function to extract YouTube video ID from URL
def extract_video_id(url):
    video_id = None
    patterns = [
        r'v=([^&]+)',  # Pattern for URLs with 'v=' parameter
        r'youtu.be/([^?]+)',  # Pattern for shortened URLs
        r'youtube.com/embed/([^?]+)'  # Pattern for embed URLs
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break
    return video_id

# Main Streamlit app
def main():
    st.title("YouTube Video Summarizer")

    # User input for YouTube video URL
    video_url = st.text_input("Enter YouTube Video URL:", "")

    # User customization options
    max_summary_length = st.slider("Max Summary Length:", 1000, 20000, 50000)

    if st.button("Summarize"):
        try:
            # Extract video ID from URL
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please enter a valid URL.")
                return

            # Get transcript of the video
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if not transcript:
                st.error("Transcript not available for this video.")
                return

            video_text = ' '.join([line['text'] for line in transcript])

            # Summarize the transcript
            summary = summarize_text(video_text, max_length=max_summary_length)

            # Extract keywords from the transcript
            keywords = extract_keywords(video_text)

            # Perform topic modeling
            topics = topic_modeling(video_text)

            # Perform sentiment analysis using VADER
            sid = SentimentIntensityAnalyzer()
            sentiment_score = sid.polarity_scores(video_text)

            # Display summarized text, keywords, topics, and sentiment
            st.subheader("Video Summary:")
            st.write(summary)

            st.subheader("Keywords:")
            st.write(keywords)

            st.subheader("Topics:")
            st.write(f"Top Topics: {', '.join(topics)}")

            st.subheader("Sentiment Analysis:")
            st.write(f"Negative: {sentiment_score['neg']}")
            st.write(f"Neutral: {sentiment_score['neu']}")
            st.write(f"Positive: {sentiment_score['pos']}")
            st.write(f"Compound: {sentiment_score['compound']}")

        except TranscriptsDisabled:
            st.error("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            st.error("No transcript found for this video.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
