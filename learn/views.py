import os
import logging
import requests
from django.shortcuts import render
from django.core.cache import cache
from django.core.paginator import Paginator
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# SUMMARY:
# This view powers a learning platform that fetches content from the Tavily API.
# It ensures that each content category (YouTube Video, Blog Post, GitHub Repository,
# Document, and Web Article) has at least three items. If not, additional targeted
# queries are performed. At the top of the search results, a concept summary is 
# displayed to explain the various resource types and how they can enhance learning.
# -----------------------------------------------------------------------------

# Load Tavily API Key
load_dotenv()
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

logger = logging.getLogger(__name__)
MIN_ITEMS_PER_TYPE = 3

def extract_content_type(url):
    """Classify content types."""
    if "youtube.com" in url or "youtu.be" in url:
        return "YouTube Video"
    elif "medium.com" in url:
        return "Blog Post"
    elif "wikipedia.org" in url:
        return "Wikipedia Article"
    elif "github.com" in url:
        return "GitHub Repository"
    elif url.endswith(('.pdf', '.doc', '.ppt')):
        return "Document"
    else:
        return "Web Article"

def fallback_description(title, url):
    """Generate a fallback description when none is provided."""
    if "youtube.com" in url or "youtu.be" in url:
        return f"Watch this YouTube video: {title}."
    elif "github.com" in url:
        return f"Explore the GitHub repository for {title}."
    elif "medium.com" in url:
        return f"Read this Medium blog post about {title}."
    elif "wikipedia.org" in url:
        return f"Learn more about {title} on Wikipedia."
    else:
        domain = url.split('/')[2] if len(url.split('/')) > 2 else "this site"
        return f"Explore {title} on {domain}."

def fetch_content(query, num_results=15):
    """Fetch content from Tavily API and return JSON data."""
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"query": query, "num_results": num_results},
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Tavily API Error: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        logger.exception("Exception occurred during API call")
        return {}

def process_results(results, content_types, max_per_type=MIN_ITEMS_PER_TYPE):
    """Process API results and organize them by content type."""
    for result in results:
        url = result.get('url', '')
        content_type = extract_content_type(url)
        description = result.get('description')
        if not description or description == "No description available":
            description = fallback_description(result.get('title', 'No title available'), url)
        item = {
            "title": result.get('title', 'No title available'),
            "description": description,
            "url": url,
            "content_type": content_type
        }
        if content_type in content_types and len(content_types[content_type]) < max_per_type:
            content_types[content_type].append(item)
    return content_types

def get_additional_query(query, category):
    """Return a specialized query string based on content category."""
    if category == "YouTube Video":
        return f"{query} site:youtube.com"
    elif category == "Blog Post":
        return f"{query} site:medium.com"
    elif category == "GitHub Repository":
        return f"{query} site:github.com"
    elif category == "Document":
        return f"{query} filetype:pdf"
    elif category == "Web Article":
        return f"{query} -site:youtube.com -site:medium.com -site:github.com"
    else:
        return query

def ensure_min_items(query, content_types, category, additional_results=5):
    """Fetch additional items for a category if it doesn't meet the minimum requirement."""
    if len(content_types[category]) < MIN_ITEMS_PER_TYPE and query:
        additional_query = get_additional_query(query, category)
        extra_data = fetch_content(additional_query, num_results=additional_results)
        extra_results = extra_data.get('results', [])
        for result in extra_results:
            if len(content_types[category]) >= MIN_ITEMS_PER_TYPE:
                break
            url = result.get('url', '')
            ct = extract_content_type(url)
            # Ensure that the additional result matches the desired category.
            if ct != category:
                continue
            description = result.get('description')
            if not description or description == "No description available":
                description = fallback_description(result.get('title', 'No title available'), url)
            item = {
                "title": result.get('title', 'No title available'),
                "description": description,
                "url": url,
                "content_type": ct
            }
            content_types[category].append(item)
    return content_types

def learn(request):
    """
    Render the Tavily-powered learn page with diverse content types.
    Enhancements include:
      - Caching search results for 10 minutes.
      - Pagination (9 items per page).
      - Filtering by content type via GET parameter 'filter'.
      - Ensuring at least three items per category (YouTube Video, Blog Post,
        GitHub Repository, Document, and Web Article) using additional targeted queries.
      - A concept summary is displayed at the top explaining the purpose and benefits
        of each content type.
      - Extra placeholder for future interactive exercises.
    """
    query = request.GET.get('query', '').strip()
    page_number = request.GET.get('page', 1)
    filter_type = request.GET.get('filter', None)  # e.g., "YouTube Video"

    # Use cache to store search results for 10 minutes.
    cache_key = f"search_{query}"
    cached_data = cache.get(cache_key) if query else None
    if cached_data:
        data = cached_data
    else:
        if query:
            data = fetch_content(query)
            cache.set(cache_key, data, timeout=600)
        else:
            data = {}

    # Initialize content type buckets with an extra placeholder category.
    content_types = {
        "YouTube Video": [],
        "Blog Post": [],
        "GitHub Repository": [],
        "Document": [],
        "Web Article": [],
        "Interactive Exercise": [  # Placeholder for future features.
            {
                "title": "Coming Soon: Interactive Exercises",
                "description": "Engage with interactive exercises designed to test your knowledge.",
                "url": "#",
                "content_type": "Interactive Exercise"
            }
        ]
    }

    if data:
        results = data.get('results', [])
        content_types = process_results(results, content_types)
    
    # For each main category, ensure we have at least MIN_ITEMS_PER_TYPE.
    for category in ["YouTube Video", "Blog Post", "GitHub Repository", "Document", "Web Article"]:
        content_types = ensure_min_items(query, content_types, category)

    # Combine all content items into a single list for pagination/filtering.
    all_items = []
    for ctype, items in content_types.items():
        all_items.extend(items)
    
    # Apply optional filtering by content type.
    if filter_type and filter_type in content_types:
        all_items = [item for item in all_items if item["content_type"] == filter_type]

    # Pagination: 9 items per page.
    paginator = Paginator(all_items, 9)
    try:
        paged_items = paginator.get_page(page_number)
    except Exception as e:
        logger.error("Pagination error: " + str(e))
        paged_items = paginator.get_page(1)
    
    # Prepare a concept summary string explaining each resource type.
    concept_summary = (
        "Concept Summary:\n"
        "- YouTube Videos: Engaging video content that visually explains topics.\n"
        "- Blog Posts: In-depth written articles discussing key concepts.\n"
        "- GitHub Repositories: Source code examples and projects to learn by doing.\n"
        "- Documents: Structured materials (e.g., PDFs, slides) for detailed study.\n"
        "- Web Articles: Online articles providing a broad overview of topics.\n"
        "- Interactive Exercises: (Coming Soon) Hands-on challenges to apply your learning.\n"
    )

    context = {
        'query': query,
        'content_types': content_types,
        'paged_items': paged_items,
        'filter_type': filter_type,
        'paginator': paginator,
        'concept_summary': concept_summary,
    }
    return render(request, 'learn/learn.html', context)
