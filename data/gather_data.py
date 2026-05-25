import requests
import pandas as pd
import time

url = 'https://graphql.anilist.co'

query = '''
query ($page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    media(type: MANGA, countryOfOrigin: "KR", sort: POPULARITY_DESC) {
      id
      title { english romaji }
      averageScore
      popularity
      genres
      tags { name rank }
      description
      characters(role: MAIN) {
        edges { node { name { full } } }
      }
    }
  }
}
'''

headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'User-Agent': 'Manhwa-Merchant/1.0'
}

def fetch_manhwa_data(max_pages=None):  # Changed default to None
    all_data = []
    page = 1
    
    while True:  # Run an open loop
        # Absolute hard stop backup condition just in case
        if max_pages is not None and page > max_pages:
            break
            
        print(f"Fetching page {page}...")
        variables = {'page': page}
        
        response = requests.post(
            url, 
            json={'query': query, 'variables': variables}, 
            headers=headers 
        )
        
        # Polite API Pacing: Pause for 1 second after every request
        time.sleep(1)
        
        if response.status_code == 429:
            print("Rate limit hit! Sleeping for 60 seconds to reset...")
            time.sleep(60)
            continue
            
        if response.status_code != 200:
            print(f"\n[!] API Error occurred with Status Code: {response.status_code}")
            print(f"[!] Server message payload:\n{response.text}\n")
            break
            
        try:
            payload = response.json()
            data = payload['data']['Page']
        except Exception as err:
            print(f"\n[!] Failed to parse JSON data.")
            break

        # Process and flatten items
        for item in data['media']:
            edges = item['characters']['edges']
            main_char_name = edges[0]['node']['name']['full'] if (edges and len(edges) > 0) else None
            
            flat_item = {
                'id': item['id'],
                'title': item['title']['english'] or item['title']['romaji'],
                'score': item['averageScore'],
                'popularity': item['popularity'],
                'genres': "|".join(item['genres']) if item['genres'] else "",
                'tags': "|".join([t['name'] for t in item['tags'] if t['rank'] > 70]) if item['tags'] else "",
                'main_char': main_char_name,
                'description': item['description']
            }
            all_data.append(flat_item)
        
        # THE GOLDEN RULE: Check if AniList explicitly tells you another page exists
        if not data['pageInfo']['hasNextPage']:
            print("Reached the absolute end of the database!")
            break
            
        page += 1
        
    return pd.DataFrame(all_data)

# EXECUTION: Passing no arguments tells it to run until hasNextPage is False
print("Starting global database extraction...")
df = fetch_manhwa_data() 

if not df.empty:
    df.to_csv('manhwa_data_complete.csv', index=False)
    print(f"Data collection complete! Harvested {len(df)} total manhwas.")
else:
    print("Extraction failed. No files generated.")
