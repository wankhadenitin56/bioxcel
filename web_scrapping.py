import requests
from urllib.parse import urlparse, parse_qs, unquote
import json
def extract_query_from_pubmed_url(pubmed_url):
    
    parsed_url = urlparse(pubmed_url)
    query_params = parse_qs(parsed_url.query)
    search_term = query_params.get('term', [''])[0]
    decoded_query = unquote(search_term)
    return decoded_query

def fetch_pubmed_pmids_from_url(pubmed_url, retmax=10000):
    
    query = extract_query_from_pubmed_url(pubmed_url)
    
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'pubmed',
        'term': query,
        'retmode': 'json',
        'retmax': retmax,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    pmids = data.get('esearchresult', {}).get('idlist', [])
    return query, pmids

# Example PubMed search URL
pubmed_url = "https://pubmed.ncbi.nlm.nih.gov/?term=%28%22Liver+fibrosis%22%5BTitle%2FAbstract%5D%29+AND+%28Thyroid+Hormone+Receptor+Beta%5BTitle%2FAbstract%5D%29&size=200"

# Fetch PMIDs
search_query, pmid_list = fetch_pubmed_pmids_from_url(pubmed_url)

# Output
print("Search Query:", search_query)
print("Total PMIDs:", len(pmid_list))
print("PMIDs:", json.dumps(pmid_list, indent=2))

