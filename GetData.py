"""
    Fetching Data from URL to apply ML to
    We will use a readily available housing database from a tarfile
"""

import os
from six.moves import urllib
import tarfile

# URL path of tarfile
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# local dataset path to extracted data to
HOUSING_PATH = os.path.join("datasets", "housing")

def fetsch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Extract data tarfile and save to dataset path
    """
    # Create dataset path
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")

    # Retrieve data tarfile from URL and save
    urllib.request.urlretrieve(housing_url, tgz_path)

    # Extract data from tarfile
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def main():
    fetsch_housing_data()

if __name__== "__main__":
    main()
