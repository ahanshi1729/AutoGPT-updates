#!/usr/bin/env python3
"""
AD-GPT: Automated Alzheimer's Disease Information System
A modernized system for collecting, analyzing, and visualizing Alzheimer's disease news and research.
Now supports both OpenAI GPT and Google Gemini models.
"""

import os
import json
import time
import textwrap
import re
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime, timedelta
import logging

# Core libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# AI APIs
from openai import OpenAI
import google.generativeai as genai

# Search and NLP
from duckduckgo_search import DDGS
import nltk
from textblob import TextBlob
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ADGPTSystem:
    """Main AD-GPT system for Alzheimer's disease information processing with multi-LLM support."""
    
    def __init__(self, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None, 
                 default_model: Literal["openai", "gemini"] = "openai"):
        """
        Initialize the AD-GPT system with API keys for supported models.
        
        Args:
            openai_api_key: OpenAI API key
            gemini_api_key: Google Gemini API key
            default_model: Default model to use ("openai" or "gemini")
        """
        self.openai_client = None
        self.gemini_model = None
        self.default_model = default_model
        
        # Initialize OpenAI if key provided
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            logger.info("✅ OpenAI client initialized")
        
        # Initialize Gemini if key provided
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("✅ Gemini client initialized")
        
        # Validate at least one model is available
        if not self.openai_client and not self.gemini_model:
            raise ValueError("At least one API key (OpenAI or Gemini) must be provided")
        
        # Adjust default model if the preferred one isn't available
        if default_model == "openai" and not self.openai_client:
            self.default_model = "gemini"
            logger.info("🔄 Switched to Gemini as default (OpenAI not available)")
        elif default_model == "gemini" and not self.gemini_model:
            self.default_model = "openai"
            logger.info("🔄 Switched to OpenAI as default (Gemini not available)")
        
        self.workspace_dir = os.path.join(os.getcwd(), "ad_gpt_workspace")
        self.ensure_workspace()
        self.setup_nltk()
        
    def setup_nltk(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
    
    def ensure_workspace(self):
        """Create workspace directory if it doesn't exist."""
        os.makedirs(self.workspace_dir, exist_ok=True)
        for subdir in ['news', 'summaries', 'visualizations', 'data']:
            os.makedirs(os.path.join(self.workspace_dir, subdir), exist_ok=True)
    
    def get_available_models(self) -> List[str]:
        """Return list of available models."""
        models = []
        if self.openai_client:
            models.append("openai")
        if self.gemini_model:
            models.append("gemini")
        return models
    
    def search_news(self, query: str = "Alzheimer's disease", max_results: int = 20) -> List[Dict]:
        """Search for Alzheimer's disease news using DuckDuckGo."""
        logger.info(f"Searching for: {query}")
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=max_results))
            
            # Save search results
            results_file = os.path.join(self.workspace_dir, 'data', 'search_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Found {len(results)} news articles")
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def scrape_article_content(self, url: str) -> Dict[str, str]:
        """Scrape article content from a URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Try to extract title
            title_tag = soup.find('title')
            title = title_tag.get_text() if title_tag else "No title found"
            
            return {
                'title': title,
                'content': text[:5000],  # Limit content length
                'url': url
            }
        
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {'title': '', 'content': '', 'url': url}
    
    def summarize_with_llm(self, text: str, max_length: int = 150, 
                          model: Optional[Literal["openai", "gemini"]] = None) -> str:
        """
        Summarize text using the specified LLM (OpenAI GPT or Google Gemini).
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            model: Model to use ("openai" or "gemini"). Uses default if None.
        """
        # Use default model if none specified
        if model is None:
            model = self.default_model
        
        prompt = f"""
        Please provide a concise summary of this Alzheimer's disease related text in {max_length} words or less:
        
        {text[:2000]}
        
        Focus on key findings, research developments, or important news.
        """
        
        try:
            if model == "openai" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_length * 2,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            
            elif model == "gemini" and self.gemini_model:
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            
            else:
                # Fallback to available model
                available_models = self.get_available_models()
                if not available_models:
                    return "No LLM available for summarization"
                
                fallback_model = available_models[0]
                logger.warning(f"Model '{model}' not available, using '{fallback_model}'")
                return self.summarize_with_llm(text, max_length, fallback_model)
        
        except Exception as e:
            logger.error(f"LLM summarization failed with {model}: {e}")
            return "Summary unavailable"
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using TextBlob."""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction using word frequency
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def process_news_articles(self, articles: List[Dict], 
                            model: Optional[Literal["openai", "gemini"]] = None) -> List[Dict]:
        """
        Process and analyze news articles using specified LLM.
        
        Args:
            articles: List of article dictionaries
            model: Model to use for processing ("openai" or "gemini")
        """
        processed_articles = []
        
        for i, article in enumerate(articles):
            logger.info(f"Processing article {i+1}/{len(articles)} with {model or self.default_model}")
            
            # Scrape full content
            content_data = self.scrape_article_content(article.get('url', ''))
            
            # Combine title and body for analysis
            full_text = f"{article.get('title', '')} {content_data.get('content', '')}"
            
            # Generate summary using specified model
            summary = self.summarize_with_llm(full_text, model=model)
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(full_text)
            
            # Extract keywords
            keywords = self.extract_keywords(full_text)
            
            processed_article = {
                'title': article.get('title', ''),
                'url': article.get('url', ''),
                'published': article.get('date', ''),
                'source': article.get('source', ''),
                'summary': summary,
                'sentiment': sentiment,
                'keywords': keywords,
                'full_content': full_text[:1000],  # Store first 1000 chars
                'processed_by': model or self.default_model
            }
            
            processed_articles.append(processed_article)
            
            # Small delay to be respectful to servers
            time.sleep(0.5)
        
        # Save processed articles
        output_file = os.path.join(self.workspace_dir, 'data', 'processed_articles.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_articles, f, indent=2, ensure_ascii=False)
        
        return processed_articles
    
    def create_visualizations(self, articles: List[Dict]):
        """Create visualizations from processed articles."""
        if not articles:
            logger.warning("No articles to visualize")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig_dir = os.path.join(self.workspace_dir, 'visualizations')
        
        # 1. Sentiment Analysis Plot
        sentiments = [article['sentiment']['polarity'] for article in articles]
        plt.figure(figsize=(10, 6))
        plt.hist(sentiments, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        plt.title('Sentiment Distribution of Alzheimer\'s Disease News', fontsize=16)
        plt.xlabel('Sentiment Polarity (-1: Negative, 1: Positive)')
        plt.ylabel('Number of Articles')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(fig_dir, 'sentiment_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Keywords Word Cloud
        all_keywords = []
        for article in articles:
            all_keywords.extend(article['keywords'])
        
        if all_keywords:
            keyword_freq = Counter(all_keywords)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keyword_freq)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Common Keywords in Alzheimer\'s Disease News', fontsize=16)
            plt.savefig(os.path.join(fig_dir, 'keywords_wordcloud.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Source Distribution
        sources = [article['source'] for article in articles if article['source']]
        if sources:
            source_counts = Counter(sources)
            plt.figure(figsize=(12, 6))
            sources_list, counts = zip(*source_counts.most_common(10))
            plt.bar(sources_list, counts, color='lightcoral')
            plt.title('News Sources Distribution', fontsize=16)
            plt.xlabel('News Source')
            plt.ylabel('Number of Articles')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'sources_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Model Usage Distribution (new visualization)
        if any('processed_by' in article for article in articles):
            models_used = [article.get('processed_by', 'unknown') for article in articles]
            model_counts = Counter(models_used)
            
            plt.figure(figsize=(8, 6))
            models_list, counts = zip(*model_counts.items())
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models_list)]
            plt.pie(counts, labels=models_list, autopct='%1.1f%%', colors=colors)
            plt.title('Articles Processed by LLM Model', fontsize=16)
            plt.savefig(os.path.join(fig_dir, 'model_usage_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Created visualizations in {fig_dir}")
    
    def generate_comprehensive_report(self, articles: List[Dict], 
                                    model: Optional[Literal["openai", "gemini"]] = None) -> str:
        """
        Generate a comprehensive report using specified LLM.
        
        Args:
            articles: List of processed articles
            model: Model to use for report generation ("openai" or "gemini")
        """
        if not articles:
            return "No articles available for report generation."
        
        # Use default model if none specified
        if model is None:
            model = self.default_model
        
        # Prepare data for LLM analysis
        summaries = [article['summary'] for article in articles[:10]]  # Use first 10 articles
        combined_summaries = '\n\n'.join(summaries)
        
        prompt = f"""
        Based on these recent Alzheimer's disease news summaries, provide a comprehensive analysis:
        
        {combined_summaries}
        
        Please provide:
        1. Key trends and developments in Alzheimer's research
        2. Major breakthroughs or concerning findings
        3. Implications for patients and caregivers
        4. Future research directions mentioned
        5. Overall sentiment and outlook
        
        Format your response as a structured report.
        """
        
        try:
            if model == "openai" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.3
                )
                report = response.choices[0].message.content.strip()
            
            elif model == "gemini" and self.gemini_model:
                response = self.gemini_model.generate_content(prompt)
                report = response.text.strip()
            
            else:
                # Fallback to available model
                available_models = self.get_available_models()
                if not available_models:
                    return "No LLM available for report generation"
                
                fallback_model = available_models[0]
                logger.warning(f"Model '{model}' not available, using '{fallback_model}'")
                return self.generate_comprehensive_report(articles, fallback_model)
            
            # Save report
            report_file = os.path.join(self.workspace_dir, 'summaries', f'comprehensive_report_{model}.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"AD-GPT Comprehensive Report (Generated by {model.upper()})\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n\n")
                f.write(report)
            
            return report
        
        except Exception as e:
            logger.error(f"Report generation failed with {model}: {e}")
            return f"Report generation failed due to {model} API error."
    
    def compare_models(self, articles: List[Dict], sample_text: str = None) -> Dict[str, str]:
        """
        Compare outputs from both OpenAI and Gemini models on the same content.
        
        Args:
            articles: List of processed articles
            sample_text: Optional specific text to compare. If None, uses first article.
        """
        available_models = self.get_available_models()
        
        if len(available_models) < 2:
            return {"error": "Need both OpenAI and Gemini to compare models"}
        
        # Use sample text or first article
        if sample_text is None and articles:
            sample_text = f"{articles[0].get('title', '')} {articles[0].get('full_content', '')}"
        elif sample_text is None:
            sample_text = "Alzheimer's disease research shows promising new treatment approaches."
        
        results = {}
        
        # Compare summaries
        for model in available_models:
            try:
                summary = self.summarize_with_llm(sample_text, model=model)
                results[f"{model}_summary"] = summary
            except Exception as e:
                results[f"{model}_summary"] = f"Error: {e}"
        
        # Compare reports if articles available
        if articles:
            for model in available_models:
                try:
                    report = self.generate_comprehensive_report(articles[:3], model=model)
                    results[f"{model}_report"] = report[:500] + "..." if len(report) > 500 else report
                except Exception as e:
                    results[f"{model}_report"] = f"Error: {e}"
        
        # Save comparison
        comparison_file = os.path.join(self.workspace_dir, 'summaries', 'model_comparison.json')
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def run_full_analysis(self, query: str = "Alzheimer's disease breakthrough treatment research",
                         model: Optional[Literal["openai", "gemini"]] = None) -> Dict[str, Any]:
        """
        Run the complete AD-GPT analysis pipeline with specified model.
        
        Args:
            query: Search query for news articles
            model: Model to use for analysis ("openai" or "gemini")
        """
        logger.info(f"Starting AD-GPT full analysis with {model or self.default_model}...")
        
        # Step 1: Search for news
        print("🔍 Searching for Alzheimer's disease news...")
        articles = self.search_news(query)
        
        if not articles:
            return {"error": "No articles found"}
        
        # Step 2: Process articles
        print(f"📄 Processing and analyzing articles with {model or self.default_model}...")
        processed_articles = self.process_news_articles(articles, model=model)
        
        # Step 3: Create visualizations
        print("📊 Creating visualizations...")
        self.create_visualizations(processed_articles)
        
        # Step 4: Generate comprehensive report
        print(f"📋 Generating comprehensive report with {model or self.default_model}...")
        report = self.generate_comprehensive_report(processed_articles, model=model)
        
        # Step 5: Prepare results summary
        results = {
            "total_articles": len(processed_articles),
            "average_sentiment": np.mean([a['sentiment']['polarity'] for a in processed_articles]),
            "top_keywords": Counter([kw for a in processed_articles for kw in a['keywords']]).most_common(10),
            "report": report,
            "model_used": model or self.default_model,
            "available_models": self.get_available_models(),
            "workspace_dir": self.workspace_dir
        }
        
        print(f"✅ Analysis complete! Results saved in: {self.workspace_dir}")
        return results

def interactive_mode():
    """Run AD-GPT in interactive mode with model selection."""
    print("🧠 AD-GPT: Alzheimer's Disease Information System")
    print("🤖 Now supports both OpenAI GPT and Google Gemini!")
    print("=" * 60)
    
    # Get API keys
    print("\n🔑 API Key Setup:")
    print("You can provide one or both API keys. The system will use whichever you provide.")
    
    openai_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    openai_key = openai_key if openai_key else None
    
    gemini_key = input("Enter your Google Gemini API key (or press Enter to skip): ").strip()
    gemini_key = gemini_key if gemini_key else None
    
    if not openai_key and not gemini_key:
        print("❌ At least one API key is required!")
        return
    
    # Choose default model
    available_models = []
    if openai_key:
        available_models.append("openai")
    if gemini_key:
        available_models.append("gemini")
    
    if len(available_models) > 1:
        print(f"\n🎯 Available models: {', '.join(available_models)}")
        default_model = input("Choose default model (openai/gemini): ").strip().lower()
        if default_model not in available_models:
            default_model = available_models[0]
            print(f"Invalid choice. Using {default_model} as default.")
    else:
        default_model = available_models[0]
    
    try:
        ad_gpt = ADGPTSystem(
            openai_api_key=openai_key, 
            gemini_api_key=gemini_key,
            default_model=default_model
        )
        print(f"✅ AD-GPT initialized successfully! Default model: {default_model}")
        print(f"🤖 Available models: {', '.join(ad_gpt.get_available_models())}")
        
        while True:
            print("\n📋 Available commands:")
            print("1. search - Search for Alzheimer's news")
            print("2. analyze - Run full analysis")
            print("3. report - Generate comprehensive report")
            print("4. compare - Compare model outputs (requires both APIs)")
            print("5. switch - Switch default model")
            print("6. status - Show system status")
            print("7. quit - Exit the system")
            
            command = input("\nEnter your command: ").strip().lower()
            
            if command == "quit":
                print("👋 Goodbye!")
                break
            
            elif command == "status":
                print(f"\n📊 System Status:")
                print(f"Available models: {', '.join(ad_gpt.get_available_models())}")
                print(f"Default model: {ad_gpt.default_model}")
                print(f"Workspace directory: {ad_gpt.workspace_dir}")
            
            elif command == "switch":
                available = ad_gpt.get_available_models()
                if len(available) > 1:
                    print(f"Available models: {', '.join(available)}")
                    new_model = input("Choose new default model: ").strip().lower()
                    if new_model in available:
                        ad_gpt.default_model = new_model
                        print(f"✅ Default model changed to {new_model}")
                    else:
                        print("❌ Invalid model choice")
                else:
                    print("Only one model available - cannot switch")
            
            elif command == "search":
                query = input("Enter search query (or press Enter for default): ").strip()
                if not query:
                    query = "Alzheimer's disease treatment research"
                
                articles = ad_gpt.search_news(query)
                print(f"Found {len(articles)} articles")
                
                for i, article in enumerate(articles[:5]):
                    print(f"{i+1}. {article.get('title', 'No title')}")
            
            elif command == "analyze":
                query = input("Enter analysis query (or press Enter for default): ").strip()
                if not query:
                    query = "Alzheimer's disease breakthrough treatment research"
                
                # Choose model for this analysis
                available = ad_gpt.get_available_models()
                if len(available) > 1:
                    model_choice = input(f"Choose model ({'/'.join(available)}) or press Enter for default: ").strip().lower()
                    model_choice = model_choice if model_choice in available else None
                else:
                    model_choice = None
                
                results = ad_gpt.run_full_analysis(query, model=model_choice)
                
                if "error" not in results:
                    print(f"\n📊 Analysis Results:")
                    print(f"Total articles analyzed: {results['total_articles']}")
                    print(f"Average sentiment: {results['average_sentiment']:.2f}")
                    print(f"Top keywords: {', '.join([kw[0] for kw in results['top_keywords'][:5]])}")
                    print(f"Model used: {results['model_used']}")
                    print(f"Files saved in: {results['workspace_dir']}")
                else:
                    print(f"❌ Error: {results['error']}")
            
            elif command == "report":
                # Load existing data or run analysis
                data_file = os.path.join(ad_gpt.workspace_dir, 'data', 'processed_articles.json')
                if os.path.exists(data_file):
                    with open(data_file, 'r', encoding='utf-8') as f:
                        articles = json.load(f)
                    
                    # Choose model for report
                    available = ad_gpt.get_available_models()
                    if len(available) > 1:
                        model_choice = input(f"Choose model for report ({'/'.join(available)}) or press Enter for default: ").strip().lower()
                        model_choice = model_choice if model_choice in available else None
                    else:
                        model_choice = None
                    
                    report = ad_gpt.generate_comprehensive_report(articles, model=model_choice)
                    print(f"\n📋 Comprehensive Report (Generated by {model_choice or ad_gpt.default_model}):")
                    print("-" * 40)
                    print(report)
                else:
                    print("❌ No data available. Please run 'analyze' first.")
            
            elif command == "compare":
                if len(ad_gpt.get_available_models()) < 2:
                    print("❌ Both OpenAI and Gemini API keys are required for comparison")
                    continue
                
                data_file = os.path.join(ad_gpt.workspace_dir, 'data', 'processed_articles.json')
                if os.path.exists(data_file):
                    with open(data_file, 'r', encoding='utf-8') as f:
                        articles = json.load(f)
                    
                    print("🔄 Comparing model outputs...")
                    comparison = ad_gpt.compare_models(articles)
                    
                    print("\n🆚 Model Comparison Results:")
                    for key, value in comparison.items():
                        if not key.endswith('_report'):  # Show summaries, not full reports
                            print(f"\n{key.upper()}:")
                            print("-" * 30)
                            print(value[:300] + "..." if len(str(value)) > 300 else value)
                else:
                    print("❌ No data available. Please run 'analyze' first.")
            
            else:
                print("❌ Unknown command. Please try again.")
    
    except Exception as e:
        print(f"❌ Error initializing AD-GPT: {e}")

if __name__ == "__main__":
    interactive_mode()