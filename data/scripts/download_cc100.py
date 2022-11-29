'''
Helper script for scrapping the cc100 dataset from https://data.statmt.org/cc-100/

Dumps out the data into a directory called 'cc100'. Expects that a directory under 
this name does not exist.
'''

import requests 
import os
import random

from urllib.request import urlretrieve

from time import sleep
from bs4 import BeautifulSoup


CC_WEBSITE_URL = 'https://data.statmt.org/cc-100/'

DATA_DIR_NAME = 'cc100'
DATA_DIR_PATH = os.path.join(os.getcwd(), DATA_DIR_NAME)

link_href_urls = []

def download_data(link_href_url):
    sleep(random.randint(5, 10))
    href, url = link_href_url
    print(f".Downloading file: {href}")
    destination = os.path.join(DATA_DIR_PATH, href)
    urlretrieve(url, destination)
    print(f"...Finished downloading file: {href}")

def main():
    page = requests.get(CC_WEBSITE_URL)

    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("a")

    if not os.path.isdir(DATA_DIR_NAME):
        os.mkdir(DATA_DIR_NAME)

    for link in results: 
        link_href = link.get('href')

        if ('txt.xz' not in link_href):
            # the file is not a text file
            continue

        link_lng = link_href.split('.')[0]
        link_url = CC_WEBSITE_URL + link_href

        if os.path.exists(os.path.join(DATA_DIR_PATH, link_href)):
            print(f"...Skipping data for language: {link_lng} already downloaded")
            continue

        download_data((link_href, link_url))

if __name__ == '__main__':
    main()