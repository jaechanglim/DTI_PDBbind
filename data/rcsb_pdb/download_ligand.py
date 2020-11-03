import os
import glob
import sys
import time
import multiprocessing
from multiprocessing import Pool

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import numpy as np


def wait_download(protein):
    wait = True
    st = time.time()
    dir_name = f"./refined_data/{protein}"
    while wait:
        if len(os.listdir(dir_name)):
            wait = False
            done = False
            st = time.time()
            while not done:
                fn = os.listdir(dir_name)[0]
                if fn.split(".")[-1] == "crdownload":
                    fn = os.listdir(dir_name)[0]
                    time.sleep(1)
                else:
                    msg = f"{protein} download success"
                    done = True
        time.sleep(1)
        t = time.time()
        if (t - st > 5):
            msg = f"{protein} download timeout"
            wait = False
    return True, msg


def get_driver(protein):
    chromedriver = driver_path
    download_path = f"./refined_data/{protein}"

    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    prefs = {"download.default_directory": download_path}
    options.add_experimental_option("prefs", prefs)

    capabilities = DesiredCapabilities.CHROME.copy()
    capabilities["acceptSslCerts"] = True
    capabilities["acceptInsecureCerts"] = True

    driver = webdriver.Chrome(
        options=options,
        executable_path=chromedriver,
        desired_capabilities=capabilities,
    )
    return driver


def download_ligand(key):
    protein, ligand = key
    driver = get_driver(protein)

    driver.get("https://www.rcsb.org/pages/download_features")
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.ID, "ligandIdList")))
    driver.find_element_by_id("ligandIdList").clear()
    driver.find_element_by_id("ligandIdList").send_keys(ligand)
    button = driver.find_elements(By.TAG_NAME, "input")
    button = [
        b for b in button if b.get_attribute("value") == "Launch Download"
    ][0]
    button.click()

    done, msg = wait_download(protein)
    if done:
        driver.quit()
    return msg


def crawling(key):
    protein, _ = key
    print(key)
    if os.path.exists(f"./refined_data/{protein}") \
            and len(os.listdir(f"./refined_data/{protein}")):
        return
    if not os.path.exists(f"./refined_data/{protein}"):
        os.mkdir(f"./refined_data/{protein}")
    try:
        msg = download_ligand(key)
        print(msg)
        return
    except Exception as e:
        print(e)
        return


if __name__ == "__main__":
    fn, driver_path = sys.argv[1]
    if not os.path.exists("refined_data"):
        os.mkdir("refined_data")
    with open(fn, "r") as r0:
        lines = r0.readlines()
        lines = [l.split() for l in lines]

    missed = [l[0] for l in lines]
    missed = [m for m in missed \
            if not os.path.exists(f"refined_data/{m}") \
            or not len(os.listdir(f"refined_data/{m}"))]
    lines = [l for l in lines if l[0] in missed]

    for l in lines:
        crawling(l)

    # pool = Pool(4)
    # r = pool.map_async(crawling, lines)
    # r = r.get()
    # pool.close()
    # pool.join()
