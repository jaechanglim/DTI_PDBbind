from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import os
import multiprocessing
from multiprocessing import Pool
import glob
import sys
import time


def find_ligand(complex):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--remote-debugging-port=9222')
    chromedriver = "/home/udg/wonho/others/chromedriver"
    driver = webdriver.Chrome(executable_path=chromedriver, options=options)
    url = f'https://www.rcsb.org/structure/{complex}'
    driver.get(url)

    print(url+'!')
    WebDriverWait(driver, 5).until(EC.presence_of_all_elements_located((By.ID, 'smallMoleculespanel')))
    boundary = driver.find_element(By.ID, 'smallMoleculespanel')
    table = boundary.find_element(By.ID, 'LigandsMainTable')
    if not table:
        table = driver.find_element(By.ID, 'LigandsTable')
    rows = table.find_elements(By.TAG_NAME, 'tr')
    ligands = [r for r in rows if 'ligand_row' in r.get_attribute('id')]
    ligand = ligands[0].get_attribute('id').split('_')[-1]
    driver.quit()
    time.sleep(1)
    return (complex, ligand)


def run(complex):
    try:
        return find_ligand(complex)
    except Exception as e:
        print(e)
        return (complex, e)


def multiprocess(keys):
    pool = Pool(4)
    r = pool.map_async(run, keys)
    r.wait()
    pool.close()
    pool.join()
    data = r.get()
    return data

# print(run('1w6o'))
# exit(-1)
fn = sys.argv[1]
dir = fn.split('.')[0].split('/')[-1]
with open(fn, 'r') as r:
    keys = r.readlines()
    keys = [key.split()[0] for key in keys]
st = time.time()
data = []
length = int(len(keys) / 40)
for i in range(1, 40):
    st = length * i
    end = length * (i+1) if i != 40 else -1
    # partial_data = multiprocess(keys[st:end])
    # data += multiprocess(keys[st:end])
    partial_data = []
    for key in keys[st:end]:
        partial_data.append(run(key))
        time.sleep(0.5)
    data += partial_data
    with open(f'./{dir}/ligand_to_complex{i}.txt', 'w') as f:
        for d in partial_data:
            complex, ligand = d
            f.write(f'{complex}\t{ligand}\n')
    time.sleep(10)
et = time.time()
print(f'{dir} end: ', et - st)
with open(f'./{dir}/ligand_to_complex.txt', 'w') as f:
    for d in data:
        complex, ligand = d
        f.write(f'{complex}\t{ligand}\n')
