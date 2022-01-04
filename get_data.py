import os
import tarfile
import urllib.request

# DOWNlOAD_ROOT variable holds the string of the download
# link for the machine learning data
DOWNLOAD_ROOT = ''
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/ housing.tgz"

#location to store dataset
HOUSING_PATH = os.path.join("datasets","housing")


def fetch_data(housing_url= HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    
    # extracting file from the tgz file
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()