'''
Helper script for scrapping the cc100 dataset from https://data.statmt.org/cc-100/

Dumps out the data into a directory called 'cc100'. Expects that a directory under 
this name does not exist.
'''

import requests 
import os
import random

from requests.auth import HTTPBasicAuth
from multiprocessing.dummy import Pool 
from bs4 import BeautifulSoup


# global variables 
WEBSITE_URL = 'https://oscar-prive.huma-num.fr/2109/packaged/'
WEBSITE_AUTH = HTTPBasicAuth('rdmartinez', '!PZFkstTwbuNmHuPr82u7!8Yn')

DATA_DIR_NAME = 'oscar'
DATA_DIR_PATH = os.path.join(os.getcwd(), DATA_DIR_NAME)

NUM_THREADS = 20

BREAK_POINT = 1e5 #set to big number to run all

def download_file(downlink_link_path):
    download_link, destination_path = downlink_link_path
    print(f"... Starting download of {download_link}")
    with requests.get(download_link, stream = True, auth = WEBSITE_AUTH) as response:
        with open(destination_path, 'wb') as fout: 
            for chunk in response.iter_content(chunk_size=8192):
                fout.write(chunk)
    print(f"xxxxxxxx Finished download of {download_link}")

def main():
    download_counter = 0

    download_links = []
    download_destinations = []

    # extracting sub-folders for all the languages in the dataset
    root_page = requests.get(WEBSITE_URL, auth=WEBSITE_AUTH)

    root_soup = BeautifulSoup(root_page.content, "html.parser")
    root_results = root_soup.find_all("a")

    if not os.path.isdir(DATA_DIR_NAME):
        os.mkdir(DATA_DIR_NAME)

    # iterating over the folders of languages 
    for link in root_results: 
        href = link.get('href')
        lng = href[:-1]
        # basic check to make sure the language is valid
        if not lng.isalnum(): 
            continue
        
        # file to save the data to locally
        local_download_folder = os.path.join(DATA_DIR_NAME, lng)
        if not os.path.exists(local_download_folder):
            os.mkdir(local_download_folder)
            print(f"Creating folder: {local_download_folder} for new language: {lng}")

        # finding download links for the specific language 
        lng_subfolder_url = WEBSITE_URL + href

        sub_page = requests.get(lng_subfolder_url, auth=WEBSITE_AUTH)
        sub_soup = BeautifulSoup(sub_page.content, "html.parser")
        sub_a_results = sub_soup.find_all("a")
        
        for sub_link in sub_a_results: 
            sub_href = sub_link.get("href")
            if ".txt.gz" in sub_href:
                # The final download link and associated file location 
                download_link = lng_subfolder_url + sub_href
                download_destination = os.path.join(local_download_folder, sub_href)

                if os.path.exists(download_destination):
                    # we've already downloaded this file
                    continue 

                if download_link not in download_links:
                    download_links.append(download_link)
                    download_destinations.append(download_destination)
                    print(f" ~~~~ Adding download link to pool: {download_link} ~~~~")
                    download_counter += 1

                    if download_counter > BREAK_POINT:
                        break
        
        if download_counter > BREAK_POINT:
            break

    print(f"\n\nLaunching thread pool with {NUM_THREADS} threads!")
    thread_pool = Pool(NUM_THREADS)
    result = thread_pool.map(download_file, list(zip(download_links, download_destinations)))

if __name__ == '__main__':
    main()