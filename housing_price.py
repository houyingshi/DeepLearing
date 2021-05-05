import os
import tarfile
import urllib
download_root = 'https://raw.githubusercontent.com/ageron/handson-ml2/master'
housing_path = os.path.join('data', 'housing')
housing_url = download_root + 'datasets/housing/housing.tgz'


def fetch_housing_data(housing_url, housing_path):
    os.makedirs(housing_path, exist_ok=True)
    file_save_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, file_save_path)
    f = tarfile.open(file_save_path)
    f.extractall(path=housing_path)
    f.close()


fetch_housing_data(housing_url, housing_path)