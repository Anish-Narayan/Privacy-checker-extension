# /home/anish/Privacy checker extension/scripts/feature_extraction.py

import pandas as pd
from urllib.parse import urlparse

# Feature extraction function
def extract_features(url):
    parsed_url = urlparse(url)
    return {
        "url": url,
        "length": len(url),
        "contains_utm": int("utm" in url),
        "contains_track": int("track" in url or "ad" in url),
        "contains_query_params": int("?" in url),
        "num_subdomains": parsed_url.netloc.count("."),
        "path_length": len(parsed_url.path),
        "contains_google": int("google" in url),
        "contains_facebook": int("facebook" in url),
        "contains_doubleclick": int("doubleclick" in url),
        "contains_analytics": int("analytics" in url or "ga" in url),
    }

# Process all URLs and save features to CSV
def save_features_to_csv():
    input_file = "../data/urls.csv"
    output_file = "../data/features.csv"

    df = pd.read_csv(input_file, names=["url", "label"])
    
    all_data = []
    for _, row in df.iterrows():
        features = extract_features(row['url'])
        features["label"] = row['label']
        all_data.append(features)

    feature_df = pd.DataFrame(all_data)
    feature_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")


if __name__ == "__main__":
    save_features_to_csv()
