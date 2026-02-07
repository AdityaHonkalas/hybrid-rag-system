import requests
from bs4 import BeautifulSoup
import random
import time
import json

def generate_random_wiki_urls_scraping(count=300):
    """
    Generate random Wikipedia URLs via web scraping
    """

    print("Scaping Wikipedia for random article URLs...\n")

    urls = set()
    
    # Add headers to avoid 403 Forbidden error
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Method 1: Scrape Wikipedia's Special:Random page multiple times
    while len(urls) < count:
        try:
            # Get a random Wikipedia page
            response = requests.get(
                "https://en.wikipedia.org/wiki/Special:Random",
                headers=headers,
                allow_redirects=True,
                timeout=5
            )
            print("Response Status Code:", response.status_code)
            if response.status_code == 200:
                # Extract the article title from the URL
                url = response.url
                urls.add(url)
                print("URL received: ", url)
                # Rate limiting to avoid being blocked
                if len(urls) % 50 == 0:
                    print(f"Generated {len(urls)} URLs...")
                    time.sleep(1)  # Wait 1 second every 50 requests
        
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            time.sleep(2)

    return list(urls)

# Alternative Method 2: Scrape category pages for article links
def generate_wiki_urls_from_categories(count=300):
    """
    Scrape Wikipedia category pages to get article URLs
    """
    print("Scraping random wikipedia URLs from category pages...\n")
    print("Random generated URLs :\n")
    
    urls = set()
    
    # Add headers to avoid 403 Forbidden error
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    categories = [
        "https://en.wikipedia.org/wiki/Category:Articles",
        "https://en.wikipedia.org/wiki/Category:Main_topic_classifications",
    ]
    
    for category_url in categories:
        if len(urls) >= count:
            break
            
        try:
            response = requests.get(category_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find all article links
            links = soup.find_all("a", href=True)
            
            for link in links:
                if len(urls) >= count:
                    break
                href = link["href"]
                
                # Filter for article links (not categories, talk pages, etc.)
                if "/wiki/" in href and "Category:" not in href and "Talk:" not in href:
                    full_url = "https://en.wikipedia.org" + href
                    urls.add(full_url)
                    print(full_url)
            time.sleep(1)  # Rate limiting
            
        except requests.exceptions.RequestException as e:
            print(f"Error scraping {category_url}: {e}")
        
    
    return list(urls)[:count]


if __name__ == "__main__":
    
    # Generate random Wikipedia URLs via scraping
    #random_wiki_urls = generate_wiki_urls_from_categories(count=10)
    random_wiki_urls = generate_random_wiki_urls_scraping(count=10)
    # Create a new file random_wiki_urls.json and save URLs
    with open("random_wiki_urls.json", "w") as f:
        json.dump({"random_wikipedia_urls": random_wiki_urls}, f, indent=4)
    
    # Close file connection
    f.close()
