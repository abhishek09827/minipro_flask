# file: app.py
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import asyncio
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from flask_cors import CORS
import json

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Initialize Chat Model
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Prompts
sentisysPrompt = '''You are a sentiment analysis expert who can assign sentiment tags to user reviews. Analyze the sentiment of the following review: "{{text}}". You have to label the review as positive, negative, or neutral. The output should be a JSON with fields "category" and "reason". The "category" field should be one of "positive", "negative", "neutral". The "reason" field should be a string explaining the reason for the category. 

Review: {text}

Output:'''

entityTaggingPromptContent = '''You are an expert at labeling a given Instagram Review as bug, feature_request, question, or feedback. You are given a review provided by a user for the app Instagram. You have to label the review as bug, feature_request, question, or feedback. The output should be a JSON with fields "category" and "reason". The "category" field should be one of "bug", "feature_request", "question", or "feedback". The "reason" field should be a string explaining the reason for the category. 

Review: {text}

Output:'''

intentClassificationPromptContent = '''You are an intent classification expert. Classify the intent of the following text: "{text}" into urgent, low, and medium category label. The output should be a JSON with fields "category" and "reason". The "category" field should include the intent. The "reason" field should be a string explaining the reason for the category. 

Review: {text}

Output:'''

# ChatPromptTemplates
sentichatPrompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(sentisysPrompt),
    HumanMessagePromptTemplate.from_template(''),
])

entityTaggingChatPrompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(entityTaggingPromptContent),
    HumanMessagePromptTemplate.from_template(''),
])

intentClassificationChatPrompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(intentClassificationPromptContent),
    HumanMessagePromptTemplate.from_template(''),
])

# Endpoints
@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        text = data['text']
        
        # Sentiment Analysis
        chain = LLMChain(llm=chat_model, prompt=sentichatPrompt)
        sentiment_result = chain.predict(text=text)
        
        # Entity Tagging
        chain = LLMChain(llm=chat_model, prompt=entityTaggingChatPrompt)
        entity_tagging_result = chain.predict(text=text)
        
        # Intent Classification
        chain = LLMChain(llm=chat_model, prompt=intentClassificationChatPrompt)
        intent_classification_result = chain.predict(text=text)
        
        combined_result = {
            'sentiment_analysis': sentiment_result,
            'entity_tagging': entity_tagging_result,
            'intent_classification': intent_classification_result
        }
        
        categories = []
        for value in combined_result.values():
            json_string = value.replace('```json\n', '').replace('\n```', '')
            json_object = json.loads(json_string)
            categories.append(json_object['category'])
        
        return jsonify(categories)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_reviews', methods=['POST'])
def process_reviews():
    try:
        reviews = request.json.get('reviews', [])
        combined_review = " ".join(reviews)
        
        insights_prompt = PromptTemplate.from_template(
            f"Based on the following user reviews: {combined_review}, generate actionable insights in brief and avoid any duplication."
        )
        chain = LLMChain(llm=chat_model, prompt=insights_prompt)
        insights = chain.predict(text=combined_review)
        
        combined_results = {
            "insights": insights
        }
        
        return jsonify(combined_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/trends_review', methods=['POST'])
def trends_review():
    try:
        reviews = request.json.get('reviews', [])
        combined_review = " ".join(reviews)
        
        trends_prompt = PromptTemplate.from_template(
            f"Analyze the following array of user reviews for trends: {combined_review}. Please keep it brief and return your answer in a .md format"
        )
        chain = LLMChain(llm=chat_model, prompt=trends_prompt)
        trends = chain.predict(text=combined_review)
        
        combined_results = {
            "trends": trends
        }
        
        return jsonify(combined_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summary_review', methods=['POST'])
def summary_review():
    try:
        reviews = request.json.get('reviews', '')
        
        summary_prompt = PromptTemplate.from_template(
            f"Summarize the following text: {reviews}. Please keep it brief and return your answer in a .md format (Markdown Format)"
        )
        chain = LLMChain(llm=chat_model, prompt=summary_prompt)
        summary = chain.predict(text=reviews)
        
        combined_results = {
            "summary": summary
        }
        
        return jsonify(combined_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
