"""
Wikipedia Q&A Pair Generator
Fetches content from Wikipedia URLs and generates question-answer pairs
"""

import json
import random
import re
import time
from typing import List, Dict
from pathlib import Path
import requests
from bs4 import BeautifulSoup


class WikipediaQAGenerator:
    def __init__(self, urls_file: str):
        """Initialize with URLs file"""
        self.urls_file = Path(urls_file)
        self.urls = []
        self.articles = []
        self.load_urls()
        
    def load_urls(self):
        """Load Wikipedia URLs from JSON file"""
        with open(self.urls_file, 'r') as f:
            data = json.load(f)
            self.urls = data.get('fixed_wiki_urls', [])
        print(f"Loaded {len(self.urls)} Wikipedia URLs")
    
    def fetch_wikipedia_content(self, url: str) -> Dict:
        """Fetch and parse Wikipedia article content"""
        try:
            # Add delay to be respectful to Wikipedia servers
            time.sleep(0.5)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1', {'id': 'firstHeading'})
            title = title_elem.text.strip() if title_elem else url.split('/')[-1].replace('_', ' ')
            
            # Extract main content paragraphs
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return None
                
            paragraphs = content_div.find_all('p')
            text_content = []
            
            for p in paragraphs[:10]:  # Get first 10 paragraphs
                text = p.get_text().strip()
                if len(text) > 50:  # Filter out very short paragraphs
                    text_content.append(text)
            
            if not text_content:
                return None
                
            return {
                'title': title,
                'url': url,
                'content': ' '.join(text_content[:5]),  # First 5 substantial paragraphs
                'source_id': title.replace(' ', '_')
            }
            
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None
    
    def fetch_all_articles(self, max_articles: int = None):
        """Fetch content from all Wikipedia URLs"""
        print("Fetching Wikipedia articles...")
        urls_to_fetch = self.urls[:max_articles] if max_articles else self.urls
        
        for i, url in enumerate(urls_to_fetch, 1):
            print(f"Fetching {i}/{len(urls_to_fetch)}: {url}")
            article = self.fetch_wikipedia_content(url)
            if article:
                self.articles.append(article)
        
        print(f"Successfully fetched {len(self.articles)} articles")
    
    def extract_key_facts(self, text: str, title: str) -> List[str]:
        """Extract key facts from text"""
        facts = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Look for informative sentences (not too short, contains useful patterns)
            if len(sentence) > 30 and any(word in sentence.lower() for word in 
                                         ['is', 'was', 'are', 'were', 'known', 'called', 
                                          'located', 'built', 'founded', 'developed', 
                                          'discovered', 'invented', 'created']):
                facts.append(sentence)
        
        return facts[:5]  # Return up to 5 key facts
    
    def generate_factual_qa(self, article: Dict) -> List[Dict]:
        """Generate factual Q&A pairs from an article"""
        qa_pairs = []
        title = article['title']
        content = article['content']
        source_id = article['source_id']
        
        # Extract key facts
        facts = self.extract_key_facts(content, title)
        
        # Generate different types of factual questions
        templates = [
            {
                'pattern': r'is (a|an) (.+?)(?:\.|,)',
                'question': f'What is {title}?',
            },
            {
                'pattern': r'located (?:in|at|on) (.+?)(?:\.|,)',
                'question': f'Where is {title} located?',
            },
            {
                'pattern': r'(?:was )?built (?:in|during|from) (.+?)(?:\.|,)',
                'question': f'When was {title} built?',
            },
            {
                'pattern': r'known (?:as|for) (.+?)(?:\.|,)',
                'question': f'What is {title} known for?',
            },
            {
                'pattern': r'founded (?:in|by) (.+?)(?:\.|,)',
                'question': f'When was {title} founded?',
            },
        ]
        
        # Try to generate Q&A from patterns
        for template in templates:
            match = re.search(template['pattern'], content, re.IGNORECASE)
            if match:
                # Find the sentence containing this match
                for fact in facts:
                    if template['pattern'].split('(')[0].strip() in fact.lower():
                        qa_pairs.append({
                            'question': template['question'],
                            'answer': fact.strip(),
                            'source_ids': [source_id],
                            'category': 'Factual'
                        })
                        break
        
        # Generate simple what-is question if no specific patterns matched
        if not qa_pairs and facts:
            first_sentence = facts[0]
            qa_pairs.append({
                'question': f'What is {title}?',
                'answer': first_sentence,
                'source_ids': [source_id],
                'category': 'Factual'
            })
        
        return qa_pairs
    
    def generate_simple_qa(self, article: Dict) -> Dict:
        """Generate a simple Q&A pair from an article"""
        title = article['title']
        content = article['content']
        source_id = article['source_id']
        
        # Get first meaningful sentence as answer
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if len(s.strip()) > 30]
        
        if not sentences:
            return None
        
        answer = sentences[0]
        
        # Determine question based on content patterns
        question_templates = [
            f'What is {title}?',
            f'Tell me about {title}.',
            f'What can you tell me about {title}?',
            f'Describe {title}.',
        ]
        
        # Try to make more specific questions based on content
        if 'located' in answer.lower() or 'in' in answer.lower():
            question_templates.insert(0, f'Where is {title} located?')
        if 'built' in answer.lower() or 'constructed' in answer.lower():
            question_templates.insert(0, f'When was {title} built?')
        if 'branch' in answer.lower() or 'field' in answer.lower() or 'studies' in answer.lower():
            question_templates.insert(0, f'What does {title} study?')
        if 'country' in answer.lower() or 'nation' in answer.lower():
            question_templates.insert(0, f'What is {title}?')
        
        return {
            'question': question_templates[0],
            'answer': answer,
            'source_ids': [source_id],
            'category': 'Factual'
        }
    
    def generate_qa_pairs(self, num_pairs: int = 100) -> List[Dict]:
        """Generate specified number of Q&A pairs"""
        print(f"\nGenerating {num_pairs} Q&A pairs...")
        
        qa_pairs = []
        used_articles = set()
        
        # Shuffle articles for variety
        shuffled_articles = random.sample(self.articles, len(self.articles))
        
        for article in shuffled_articles:
            if len(qa_pairs) >= num_pairs:
                break
            
            # Generate Q&A from this article
            qa = self.generate_simple_qa(article)
            
            if qa and article['source_id'] not in used_articles:
                qa_pairs.append(qa)
                used_articles.add(article['source_id'])
        
        # If we need more, generate additional variations
        if len(qa_pairs) < num_pairs:
            print(f"Generated {len(qa_pairs)} unique pairs, creating variations to reach {num_pairs}...")
            
            while len(qa_pairs) < num_pairs and shuffled_articles:
                article = random.choice(shuffled_articles)
                qa = self.generate_simple_qa(article)
                
                if qa:
                    # Vary the question slightly
                    variations = [
                        f'What is {article["title"]}?',
                        f'Can you explain what {article["title"]} is?',
                        f'Tell me about {article["title"]}.',
                        f'Describe {article["title"]}.',
                    ]
                    qa['question'] = random.choice(variations)
                    qa_pairs.append(qa)
        
        # Limit to requested number
        qa_pairs = qa_pairs[:num_pairs]
        
        # Add IDs
        for i, qa in enumerate(qa_pairs, 1):
            qa['id'] = i
        
        print(f"Successfully generated {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    
    def save_qa_dataset(self, qa_pairs: List[Dict], output_file: str = None):
        """Save Q&A pairs in the specified format"""
        if not output_file:
            output_file = "wikipedia_qa_100.json"
        
        output_data = {
            "dataset": "wikipedia_qa_100",
            "data": qa_pairs
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Generated {len(qa_pairs)} Q&A pairs")
        print(f"✓ Saved to {output_path.absolute()}")
        print(f"{'='*60}")
        
        # Show sample
        if qa_pairs:
            print("\nSample Q&A pairs:")
            for i, qa in enumerate(qa_pairs[:3], 1):
                print(f"\n{i}. Question: {qa['question']}")
                print(f"   Answer: {qa['answer'][:100]}...")
                print(f"   Source: {qa['source_ids'][0]}")
        
        return output_path


def main():
    """Main execution function"""
    # Configuration
    urls_file = "200_fixed_urls.json"
    output_file = "wikipedia_qa_100.json"
    num_qa_pairs = 100
    max_articles_to_fetch = 150  # Fetch more than needed for variety
    
    print("="*60)
    print("Wikipedia Q&A Dataset Generator")
    print("="*60)
    
    # Initialize generator
    generator = WikipediaQAGenerator(urls_file)
    
    # Fetch Wikipedia articles
    generator.fetch_all_articles(max_articles=max_articles_to_fetch)
    
    if not generator.articles:
        print("Error: No articles were successfully fetched!")
        return
    
    # Generate Q&A pairs
    qa_pairs = generator.generate_qa_pairs(num_pairs=num_qa_pairs)
    
    # Save to JSON
    output_path = generator.save_qa_dataset(qa_pairs, output_file)
    
    print(f"\nDone! Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
